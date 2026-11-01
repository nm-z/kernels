# gemvb.jl: one matrix phase of layer l (slots j0 .. j0 + nm - 1) at decode width, R times with a grid barrier
# between, on die 0; its time a run and GB/s from the phase accounting. In the server:
#   eval include("prof/gemvb.jl"); gemvb(0, 5, 2)
function k_gemvb!(A::KA, l::Int32, j0::Int32, nm::Int32, R::Int32)
    for r in Int32(0):R-Int32(1)
        gemv_ph!(A, l, j0, nm, r, Val(1))
        gbar!(A)
    end
    return
end
function gemvb(l, j0, nm; R = 8)
    PE = PK[]; K = PE.K; AMDGPU.device!(SRV[].dies[1].dev)
    acct = AMDGPU.zeros(UInt64, 3 + 2R); ah = zeros(UInt64, 3 + 2R); ah[4:2:end] .= typemax(UInt64); copyto!(acct, ah)
    copyto!(K[1].bar, zeros(UInt32, length(K[1].bar)))
    A = kargs(K, PE.hx, 1, 0, acct, AMDGPU.zeros(Int32, 1), 0, 0, PE.nwg, nothing, PE.npos, 0)
    @roc groupsize = 256 gridsize = PE.nwg maxthreads = 256 waves_per_eu = KWPE[] k_gemvb!(A, Int32(l), Int32(j0), Int32(nm), Int32(R))
    AMDGPU.synchronize()
    h = Array(acct); us = [(h[5+2r] - h[4+2r]) / PE.hz * 1e6 for r in 0:R-1]
    L = SRV[].dies[1].L[l+1]; ms = (L.m1, L.m2, L.m3, L.m4, L.out, L.gate, L.up, L.down); by = sum(sizeof(ms[j0+i].W) + sizeof(ms[j0+i].D) for i in 1:nm)
    AMDGPU.unsafe_free!(acct)
    m = sort(us)[cld(R, 2)]
    (round.(us, digits = 1), round(by / 1e6, digits = 2), round(by / m / 1e3, digits = 1))   # us a run, MB, GB/s (median)
end
# the same with ~64 KB of other code run between the repeats (the instruction cache evicted)
@inline tie(a::Float32, t::UInt64) = Base.llvmcall("""
    %r = call float asm sideeffect "; tie \$0 \$2", "=v,0,s"(float %0, i64 %1)
    ret float %r""", Float32, Tuple{Float32,UInt64}, a, t)
@inline function evict!(A::KA)
    a = ntuple(i -> tie(Float32(i), memtime()), Val(8))
    Base.Cartesian.@nexprs 1024 j -> begin
        a = let a = a; ntuple(i -> fma(a[i], 1.0001f0, 0.5f0), Val(8)) end
    end
    s = a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7] + a[8]
    s == 12345.0f0 && st!(A.red, s, 0)
end
function k_gemvc!(A::KA, l::Int32, j0::Int32, nm::Int32, R::Int32)
    for r in Int32(0):R-Int32(1)
        gemv_ph!(A, l, j0, nm, r, Val(1))
        evict!(A)
        gbar!(A)
    end
    return
end
function gemvc(l, j0, nm; R = 8)
    PE = PK[]; K = PE.K; AMDGPU.device!(SRV[].dies[1].dev)
    acct = AMDGPU.zeros(UInt64, 3 + 2R); ah = zeros(UInt64, 3 + 2R); ah[4:2:end] .= typemax(UInt64); copyto!(acct, ah)
    copyto!(K[1].bar, zeros(UInt32, length(K[1].bar)))
    A = kargs(K, PE.hx, 1, 0, acct, AMDGPU.zeros(Int32, 1), 0, 0, PE.nwg, nothing, PE.npos, 0)
    @roc groupsize = 256 gridsize = PE.nwg maxthreads = 256 waves_per_eu = KWPE[] k_gemvc!(A, Int32(l), Int32(j0), Int32(nm), Int32(R))
    AMDGPU.synchronize()
    h = Array(acct); us = [(h[5+2r] - h[4+2r]) / PE.hz * 1e6 for r in 0:R-1]
    AMDGPU.unsafe_free!(acct)
    round.(us, digits = 1)
end
# control: a small block (about 2 KB of code, the same delay shape) between the repeats
@inline function small!(A::KA)
    a = ntuple(i -> tie(Float32(i), memtime()), Val(8))
    for it in Int32(1):opq(Int32(32))
        Base.Cartesian.@nexprs 32 j -> begin
            a = let a = a; ntuple(i -> fma(a[i], 1.0001f0, 0.5f0), Val(8)) end
        end
    end
    s = a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7] + a[8]
    s == 12345.0f0 && st!(A.red, s, 0)
end
function k_gemvs!(A::KA, l::Int32, j0::Int32, nm::Int32, R::Int32)
    for r in Int32(0):R-Int32(1)
        gemv_ph!(A, l, j0, nm, r, Val(1))
        small!(A)
        gbar!(A)
    end
    return
end
function gemvs(l, j0, nm; R = 8)
    PE = PK[]; K = PE.K; AMDGPU.device!(SRV[].dies[1].dev)
    acct = AMDGPU.zeros(UInt64, 3 + 2R); ah = zeros(UInt64, 3 + 2R); ah[4:2:end] .= typemax(UInt64); copyto!(acct, ah)
    copyto!(K[1].bar, zeros(UInt32, length(K[1].bar)))
    A = kargs(K, PE.hx, 1, 0, acct, AMDGPU.zeros(Int32, 1), 0, 0, PE.nwg, nothing, PE.npos, 0)
    @roc groupsize = 256 gridsize = PE.nwg maxthreads = 256 waves_per_eu = KWPE[] k_gemvs!(A, Int32(l), Int32(j0), Int32(nm), Int32(R))
    AMDGPU.synchronize()
    h = Array(acct); us = [(h[5+2r] - h[4+2r]) / PE.hz * 1e6 for r in 0:R-1]
    AMDGPU.unsafe_free!(acct)
    round.(us, digits = 1)
end
