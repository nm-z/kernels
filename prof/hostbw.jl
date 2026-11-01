# hostbw.jl: a die reading page-locked host memory (the exchange's medium): n bytes by every thread of the
# persistent kernel's grid, 8-byte loads; and one thread's dependent-load latency. In the server:
#   eval include("prof/hostbw.jl"); hostbw()
function k_hrd!(o, p::LP{UInt64}, n8::Int32)
    i = Int32(256) * (Int32(workgroupIdx().x) - Int32(1)) + Int32(workitemIdx().x) - Int32(1); s = UInt64(0)
    nt = Int32(256) * Int32(gridGroupDim().x)
    t0 = memtime()
    while i < n8
        s += ld64a_sys(p + 8Int64(i)); i += nt
    end
    s == UInt64(12345) && unsafe_store!(pointer(o), s, 3)
    (workgroupIdx().x == 1 && workitemIdx().x == 1) && unsafe_store!(pointer(o), memtime() - t0, 1)
    return
end
function k_hlat!(o, p::LP{UInt64}, n::Int32)
    j = UInt64(0); t0 = memtime()
    for _ in Int32(1):n; j = ld64a_sys(p + 8Int64(j & 0xff)); end
    unsafe_store!(pointer(o), memtime() - t0, 1); unsafe_store!(pointer(o), j, 2)
    return
end
function hostbw()
    PE = PK[]; AMDGPU.device!(SRV[].dies[1].dev); o = AMDGPU.zeros(UInt64, 3)
    p = reinterpret(LP{UInt64}, PE.hx.p + PE.hx.out - 64 << 20)       # 64 MB below the counters: exchange slots, idle now
    r = []
    for kb in (40, 160, 1024, 8192)
        n8 = Int32(kb * 1024 ÷ 8)
        for _ in 1:2
            @roc groupsize = 256 gridsize = PE.nwg k_hrd!(o, p, n8); AMDGPU.synchronize()
        end
        t = Array(o)[1] / PE.hz
        push!(r, (kb, round(t * 1e6, digits = 1), round(kb * 1024 / t / 1e9, digits = 2)))   # KB, us, GB/s
    end
    @roc k_hlat!(o, p, Int32(200)); AMDGPU.synchronize(); @roc k_hlat!(o, p, Int32(200)); AMDGPU.synchronize()
    push!(r, ("latency us", round(Array(o)[1] / PE.hz / 200 * 1e6, digits = 2)))
    AMDGPU.unsafe_free!(o); r
end
# four dies reading at once, from a fresh pinned buffer with the given hipHostMalloc flags (each die its own 8 MB)
function hostbw4(flags; mb = 8)
    PE = PK[]; n = mb << 20; r = Ref{Ptr{Cvoid}}(C_NULL)
    AMDGPU.HIP.hipHostMalloc(r, 4n, flags); Libc.memset(Ptr{UInt8}(r[]), 1, 4n)
    os = [(AMDGPU.device!(D.dev); AMDGPU.zeros(UInt64, 3)) for D in SRV[].dies]
    res = Float64[]
    for rep in 1:2
        t0 = time_ns()
        for (i, D) in enumerate(SRV[].dies)
            AMDGPU.device!(D.dev); p = reinterpret(LP{UInt64}, Ptr{UInt8}(r[]) + (i - 1) * n)
            @roc groupsize = 256 gridsize = PE.nwg k_hrd!(os[i], p, Int32(n ÷ 8))
        end
        for D in SRV[].dies; AMDGPU.device!(D.dev); AMDGPU.synchronize(); end
        rep == 2 && push!(res, 4n / ((time_ns() - t0) / 1e9) / 1e9)
    end
    for (o, D) in zip(os, SRV[].dies); AMDGPU.device!(D.dev); AMDGPU.unsafe_free!(o); end
    AMDGPU.HIP.hipHostFree(r[])
    round(res[1], digits = 2)                              # GB/s, all four dies together
end
# the same bandwidth test, then a visibility test: die 0 writes a tagged word stream, die 1 polls it
function hostvis(flags; n = 2560)
    r = Ref{Ptr{Cvoid}}(C_NULL); AMDGPU.HIP.hipHostMalloc(r, 8n, flags); Libc.memset(Ptr{UInt8}(r[]), 0, 8n)
    p = reinterpret(LP{UInt64}, Ptr{UInt8}(r[])); D0, D1 = SRV[].dies[1], SRV[].dies[2]
    AMDGPU.device!(D1.dev); o = AMDGPU.zeros(Float32, n)
    @roc groupsize = 256 gridsize = cld(n, 256) k_hvr!(o, p, Int32(n))
    AMDGPU.device!(D0.dev); @roc groupsize = 256 gridsize = cld(n, 256) k_hvw!(p, Int32(n)); AMDGPU.synchronize()
    AMDGPU.device!(D1.dev); AMDGPU.synchronize()
    h = Array(o); AMDGPU.unsafe_free!(o); AMDGPU.HIP.hipHostFree(r[])
    count(i -> h[i] == Float32(i - 1), 1:n)
end
function k_hvw!(p::LP{UInt64}, n::Int32)
    i = Int32(256) * (Int32(workgroupIdx().x) - Int32(1)) + Int32(workitemIdx().x) - Int32(1)
    i < n && st64_sys(p + 8Int64(i), xword(Float32(i), UInt64(7) << 32))
    return
end
function k_hvr!(o, p::LP{UInt64}, n::Int32)
    i = Int32(256) * (Int32(workgroupIdx().x) - Int32(1)) + Int32(workitemIdx().x) - Int32(1)
    if i < n
        u = ld64a_sys(p + 8Int64(i)); t0 = memtime()
        while u >> 32 != UInt64(7) && memtime() - t0 < UInt64(27_000_000); u = ld64a_sys(p + 8Int64(i)); end
        u >> 32 == UInt64(7) && unsafe_store!(pointer(o), reinterpret(Float32, UInt32(u & 0xffffffff)), i + 1)
    end
    return
end
