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
