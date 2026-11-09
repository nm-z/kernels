include(joinpath(@__DIR__, "..", "lfm3.jl"))
function kl(S::Sys, ids, refpath, n)
    t0 = time(); Lg = logits(S, ids[1:n]); tp = time() - t0
    ref = Mmap.mmap(refpath, Matrix{UInt16}, (NV, n)); kls = Float64[]; top = 0
    for p in 1:n
        r = Float64.(bf32.(view(ref, :, p))); o = Float64.(view(Lg, :, p)); (all(isfinite, r) && all(isfinite, o)) || (push!(kls, Inf); continue)
        lr = r .- (maximum(r) + log(sum(exp.(r .- maximum(r))))); lo = o .- (maximum(o) + log(sum(exp.(o .- maximum(o)))))
        push!(kls, sum(exp.(lr) .* (lr .- lo))); top += argmax(r) == argmax(o)
    end
    m = length(kls)
    @printf("KL vs BF16 over %d positions: mean %.6f, median %.6f, max %.5f; top-1 %.4f; %.1f s (%.1f tok/s)\n", m, sum(kls) / m, sort(kls)[cld(m, 2)], maximum(kls), top / m, tp, n / tp)
end
S = load(expanduser("~/models/lfm3-1.7b-base"); nslot = 0, reserve = 0.6e9, hostgb = 2.0)
ids = Int.(reinterpret(Int32, read(expanduser("~/models/lfm3-1.7b-base/ref.ids"))))
kl(S, ids, expanduser("~/models/lfm3-1.7b-base/ref.logits.bf16"), parse(Int, ARGS[1]))
