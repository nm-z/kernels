# p2p.jl: die 0 writes n words into its partner's rx1 over P2P (st32_sys); die 1 waits for them (xwait) and sums
#   eval include("prof/p2p.jl"); p2ptest(2560)
function k_p2pw!(px::LP{UInt32}, n::Int32)
    i = Int32(256) * (Int32(workgroupIdx().x) - Int32(1)) + Int32(workitemIdx().x) - Int32(1)
    i < n && st32_sys(px + 4Int64(i), xmark(Float32(i)))
    return
end
function k_p2pr!(o, rx::LP{UInt32}, n::Int32)
    i = Int32(256) * (Int32(workgroupIdx().x) - Int32(1)) + Int32(workitemIdx().x) - Int32(1)
    if i < n
        u = ld32a_sys(rx + 4Int64(i)); t0 = memtime()
        while u == UInt32(0) && memtime() - t0 < UInt64(27_000_000); u = ld32a_sys(rx + 4Int64(i)); end
        st32_sys(rx + 4Int64(i), UInt32(0))
        u != UInt32(0) && unsafe_store!(pointer(o), reinterpret(Float32, u), i + 1)
    end
    return
end
function p2ptest(n)
    PE = PK[]; D0, D1 = SRV[].dies[1], SRV[].dies[2]
    AMDGPU.device!(D1.dev); o = AMDGPU.zeros(Float32, n)
    rx = lp(UInt32, PE.K[2].rx)
    @roc groupsize = 256 gridsize = cld(n, 256) k_p2pr!(o, rx, Int32(n))
    AMDGPU.device!(D0.dev)
    @roc groupsize = 256 gridsize = cld(n, 256) k_p2pw!(rx, Int32(n)); AMDGPU.synchronize()
    AMDGPU.device!(D1.dev); AMDGPU.synchronize()
    h = Array(o); AMDGPU.unsafe_free!(o)
    (count(i -> abs(h[i] - (i - 1)) <= 1.0f-3 * max(1, i), 1:n), n)
end
# the same with the writer flushing the reader die's HDP (its MemFlushCntl register, mapped for the GPU) after its stores
function k_p2pwf!(px::LP{UInt32}, hdp::LP{UInt32}, n::Int32)
    i = Int32(256) * (Int32(workgroupIdx().x) - Int32(1)) + Int32(workitemIdx().x) - Int32(1)
    i < n && st32_sys(px + 4Int64(i), xmark(Float32(i)))
    fence_rel_sys(); sync_workgroup()
    workitemIdx().x == 1 && st32_sys(hdp, UInt32(1))
    return
end
function hdpmap(D)                                         # die D's HDP flush register, as a device pointer
    a = AMDGPU.HIP.properties(D.dev).hdpMemFlushCntl; base = Ptr{Cvoid}(UInt(a) & ~UInt(4095))
    e = try AMDGPU.HIP.hipHostRegister(base, 4096, 0x1 | 0x4) catch err; err end   # Portable | IoMemory
    dp = Ref{Ptr{Cvoid}}(C_NULL); AMDGPU.HIP.hipHostGetDevicePointer(dp, base, 0)
    (e, Ptr{UInt32}(UInt(dp[]) + (UInt(a) & UInt(4095))))
end
function p2ptestf(n)
    PE = PK[]; D0, D1 = SRV[].dies[1], SRV[].dies[2]
    e, h1 = hdpmap(D1)
    AMDGPU.device!(D1.dev); o = AMDGPU.zeros(Float32, n)
    rx = lp(UInt32, PE.K[2].rx)
    @roc groupsize = 256 gridsize = cld(n, 256) k_p2pr!(o, rx, Int32(n))
    AMDGPU.device!(D0.dev)
    @roc groupsize = 256 gridsize = cld(n, 256) k_p2pwf!(rx, lp(UInt32, h1), Int32(n)); AMDGPU.synchronize()
    AMDGPU.device!(D1.dev); AMDGPU.synchronize()
    h = Array(o); AMDGPU.unsafe_free!(o)
    (e, h1, count(i -> abs(h[i] - (i - 1)) <= 1.0f-3 * max(1, i), 1:n), n)
end
# the reader flushing its own HDP while it waits (its MemFlushCntl at the address HIP reports)
function k_p2prf!(o, rx::LP{UInt32}, hdp::LP{UInt32}, n::Int32)
    i = Int32(256) * (Int32(workgroupIdx().x) - Int32(1)) + Int32(workitemIdx().x) - Int32(1)
    if i < n
        u = ld32a_sys(rx + 4Int64(i)); t0 = memtime()
        while u == UInt32(0) && memtime() - t0 < UInt64(27_000_000)
            st32_sys(hdp, UInt32(1)); u = ld32a_sys(rx + 4Int64(i))
        end
        st32_sys(rx + 4Int64(i), UInt32(0))
        u != UInt32(0) && unsafe_store!(pointer(o), reinterpret(Float32, u), i + 1)
    end
    return
end
function p2ptestr(n)
    PE = PK[]; D0, D1 = SRV[].dies[1], SRV[].dies[2]
    h1 = AMDGPU.HIP.properties(D1.dev).hdpMemFlushCntl
    AMDGPU.device!(D1.dev); o = AMDGPU.zeros(Float32, n)
    rx = lp(UInt32, PE.K[2].rx)
    @roc groupsize = 256 gridsize = cld(n, 256) k_p2prf!(o, rx, lp(UInt32, h1), Int32(n))
    AMDGPU.device!(D0.dev)
    @roc groupsize = 256 gridsize = cld(n, 256) k_p2pw!(rx, Int32(n)); AMDGPU.synchronize()
    AMDGPU.device!(D1.dev); AMDGPU.synchronize()
    h = Array(o); AMDGPU.unsafe_free!(o)
    (count(i -> abs(h[i] - (i - 1)) <= 1.0f-3 * max(1, i), 1:n), n)
end
# die 0 writes its own fine-grained rx; die 1 polls it over P2P (reads)
function p2ptestpull(n)
    PE = PK[]; D0, D1 = SRV[].dies[1], SRV[].dies[2]
    rx0 = lp(UInt32, PE.K[1].rx)
    AMDGPU.device!(D1.dev); o = AMDGPU.zeros(Float32, n)
    @roc groupsize = 256 gridsize = cld(n, 256) k_p2pr!(o, rx0, Int32(n))
    AMDGPU.device!(D0.dev)
    @roc groupsize = 256 gridsize = cld(n, 256) k_p2pw!(rx0, Int32(n)); AMDGPU.synchronize()
    AMDGPU.device!(D1.dev); AMDGPU.synchronize()
    h = Array(o); AMDGPU.unsafe_free!(o)
    (count(i -> abs(h[i] - (i - 1)) <= 1.0f-3 * max(1, i), 1:n), n)
end
function p2psame(n)
    PE = PK[]; D1 = SRV[].dies[2]; rx = lp(UInt32, PE.K[2].rx)
    AMDGPU.device!(D1.dev); o = AMDGPU.zeros(Float32, n)
    @roc groupsize = 256 gridsize = cld(n, 256) k_p2pw!(rx, Int32(n)); AMDGPU.synchronize()
    @roc groupsize = 256 gridsize = cld(n, 256) k_p2pr!(o, rx, Int32(n)); AMDGPU.synchronize()
    h = Array(o); AMDGPU.unsafe_free!(o)
    (count(i -> abs(h[i] - (i - 1)) <= 1.0f-3 * max(1, i), 1:n), n, h[1:6])
end
# push test into a buffer of die 1 allocated with the given hipExtMallocWithFlags flag
function p2pflag(n, flag)
    PE = PK[]; D0, D1 = SRV[].dies[1], SRV[].dies[2]
    AMDGPU.device!(D1.dev); r = Ref{Ptr{Cvoid}}(C_NULL)
    AMDGPU.HIP.hipExtMallocWithFlags(r, 4n, flag); AMDGPU.HIP.hipMemset(r[], 0, 4n); AMDGPU.synchronize()
    rx = lp(UInt32, Ptr{UInt8}(r[])); o = AMDGPU.zeros(Float32, n)
    @roc groupsize = 256 gridsize = cld(n, 256) k_p2pr!(o, rx, Int32(n))
    AMDGPU.device!(D0.dev)
    @roc groupsize = 256 gridsize = cld(n, 256) k_p2pw!(rx, Int32(n)); AMDGPU.synchronize()
    AMDGPU.device!(D1.dev); AMDGPU.synchronize()
    h = Array(o); AMDGPU.unsafe_free!(o); AMDGPU.HIP.hipFree(r[])
    (Int(flag), count(i -> abs(h[i] - (i - 1)) <= 1.0f-3 * max(1, i), 1:n), n)
end
# push test from die a into an uncached buffer of die b (1-based die indices)
function p2pab(n, a, b; flag = AMDGPU.HIP.hipDeviceMallocUncached)
    Da, Db = SRV[].dies[a], SRV[].dies[b]
    AMDGPU.device!(Db.dev); r = Ref{Ptr{Cvoid}}(C_NULL)
    AMDGPU.HIP.hipExtMallocWithFlags(r, 4n, flag); AMDGPU.HIP.hipMemset(r[], 0, 4n); AMDGPU.synchronize()
    rx = lp(UInt32, Ptr{UInt8}(r[])); o = AMDGPU.zeros(Float32, n)
    @roc groupsize = 256 gridsize = cld(n, 256) k_p2pr!(o, rx, Int32(n))
    AMDGPU.device!(Da.dev)
    @roc groupsize = 256 gridsize = cld(n, 256) k_p2pw!(rx, Int32(n)); AMDGPU.synchronize()
    AMDGPU.device!(Db.dev); AMDGPU.synchronize()
    h = Array(o); AMDGPU.unsafe_free!(o); AMDGPU.HIP.hipFree(r[])
    (a - 1, b - 1, count(i -> abs(h[i] - (i - 1)) <= 1.0f-3 * max(1, i), 1:n), n)
end
