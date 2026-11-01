# lat.jl: one thread's dependent-load latency on die 0 (pointer chase), in the server:
#   eval include("prof/lat.jl"); lat(stride_words, n)
function k_chase!(o, a, n::Int32)
    i = Int32(0); t0 = memtime()
    for _ in Int32(1):n
        i = unsafe_load(pointer(a), i + 1)
    end
    unsafe_store!(pointer(o), memtime() - t0, 1); unsafe_store!(pointer(o), UInt64(i), 2)
    return
end
function lat(stride, n; len = 16 << 20)
    AMDGPU.device!(SRV[].dies[1].dev)
    h = Int32[(i + stride) % len for i in 0:len-1]; a = ROCArray(h); o = AMDGPU.zeros(UInt64, 2)
    @roc k_chase!(o, a, Int32(n)); AMDGPU.synchronize()
    @roc k_chase!(o, a, Int32(n)); AMDGPU.synchronize()
    r = Array(o)[1] / PK[].hz / n * 1e9
    AMDGPU.unsafe_free!(a); AMDGPU.unsafe_free!(o)
    round(r, digits = 1)
end
