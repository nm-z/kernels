# isa.jl: one phase compiled as its own kernel, its GCN assembly into a file. In the server:
#   eval include("prof/isa.jl"); isadump(rec_ph!, "/data/models/prof/rec.s")
function k_one!(A::KA, f::F, l::Int32, P::Int32) where {F}
    f(A, l, Int32(0), P)
    return
end
function isadump(f, out)
    K = PK[].K; hx = PK[].hx; AMDGPU.device!(SRV[].dies[1].dev)
    A = kargs(K, hx, 1, 0, AMDGPU.zeros(UInt64, 4), AMDGPU.zeros(Int32, 1), 0, 0, 1, nothing, PK[].npos, 0)
    open(out, "w") do io
        AMDGPU.@device_code_gcn io = io @roc launch = false maxthreads = 256 k_one!(A, f, Int32(0), Int32(1))
    end
    out
end
# the whole persistent kernel
function isareq(out)
    K = PK[].K; hx = PK[].hx; AMDGPU.device!(SRV[].dies[1].dev)
    A = kargs(K, hx, 1, 0, AMDGPU.zeros(UInt64, 4), AMDGPU.zeros(Int32, 1), 0, 0, 1, nothing, PK[].npos, 0)
    open(out, "w") do io
        AMDGPU.@device_code_gcn io = io @roc launch = false maxthreads = 256 waves_per_eu = KWPE[] k_req!(A)
    end
    out
end
