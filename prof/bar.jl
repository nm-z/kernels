# bar.jl: R grid barriers in a row on die 0 (every workgroup of the persistent kernel's size), the mean time of one
#   eval include("prof/bar.jl"); barus(1000)
function k_bar!(A::KA, o, R::Int32)
    t0 = memtime()
    for r in Int32(1):R; gbar!(A); end
    (wgid() == Int32(0) && tid() == Int32(0)) && unsafe_store!(pointer(o), memtime() - t0, 1)
    return
end
function barus(R)
    PE = PK[]; K = PE.K; AMDGPU.device!(SRV[].dies[1].dev)
    copyto!(K[1].bar, zeros(UInt32, length(K[1].bar))); o = AMDGPU.zeros(UInt64, 1)
    A = kargs(K, PE.hx, 1, 0, AMDGPU.zeros(UInt64, 4), AMDGPU.zeros(Int32, 1), 0, 0, PE.nwg, nothing, PE.npos, 0)
    for _ in 1:2
        @roc groupsize = 256 gridsize = PE.nwg maxthreads = 256 waves_per_eu = KWPE[] k_bar!(A, o, Int32(R)); AMDGPU.synchronize()
    end
    r = Array(o)[1] / PE.hz / R * 1e6; AMDGPU.unsafe_free!(o); round(r, digits = 2)
end
