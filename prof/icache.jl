# icache.jl: does cold code cost much on gfx900? A straight-line block of 8192 FMAs (8 independent chains, about
# 64 KB of code) run twice in one kernel; the first pass fetches the code, the second runs from the instruction cache.
# The chains are tied to the clock reads through opaque asm so the compiler keeps them between the reads.
#   HIP_VISIBLE_DEVICES=3 julia icache.jl
using AMDGPU
@inline memtime() = ccall("llvm.amdgcn.s.memrealtime", llvmcall, UInt64, ())
@inline tie(a::Float32, t::UInt64) = Base.llvmcall("""
    %r = call float asm sideeffect "; tie \$0 \$2", "=v,0,s"(float %0, i64 %1)
    ret float %r""", Float32, Tuple{Float32,UInt64}, a, t)
function k_ic!(o, x0::Float32)
    a = ntuple(i -> x0 + Float32(i), Val(8))
    for it in 1:2
        t0 = memtime()
        a = let a = a, t0 = t0; ntuple(i -> tie(a[i], t0), Val(8)) end
        Base.Cartesian.@nexprs 1024 j -> begin
            a = let a = a; ntuple(i -> fma(a[i], 1.0001f0, 0.5f0), Val(8)) end
        end
        s = a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7] + a[8]
        t1 = memtime(); s = tie(s, t1)
        workitemIdx().x == 1 && workgroupIdx().x == 1 && unsafe_store!(pointer(o), t1 - t0, it)
        a = (s, a[2], a[3], a[4], a[5], a[6], a[7], a[8])
    end
    unsafe_store!(pointer(o), reinterpret(UInt64, Float64(a[1])), 3)
    return
end
o = AMDGPU.zeros(UInt64, 3)
for g in (1, 56, 112)
    @roc groupsize = 64 gridsize = g k_ic!(o, 1.0f0); AMDGPU.synchronize()
    @roc groupsize = 64 gridsize = g k_ic!(o, 1.0f0); AMDGPU.synchronize()
    h = Array(o)
    println("grid $g: pass 1 ", round(h[1] / 27, digits = 1), " us, pass 2 ", round(h[2] / 27, digits = 1), " us")
end
