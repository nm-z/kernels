# lfm3-sub.jl: the head over a subset of the positions (rows = LOSS with PL < P) against the same rows of a full run
#   julia --project=env/cuda -t 8 test/lfm3-sub.jl DIR
include(joinpath(@__DIR__, "..", "lfm3.jl"))
S = load(ARGS[1]; nslot = 0, reserve = 0.6e9, hostgb = 2.0); P = 96
ids = Int.(reinterpret(Int32, read(joinpath(ARGS[1], "ref.ids"))))[1:P]
full = logits(S, ids); sub = logits(S, ids; lrows = Int32.(50:P-1))
d = maximum(abs.(full[:, 51:P] .- sub)); @printf("logits rows 50..95: max |full - sub| %.3g (|full| max %.3g)\n", d, maximum(abs.(full)))
for c in 1:4; q = full[:, 50 + c]; w = sub[:, c]; @printf("  row %d: argmax full %d sub %d, max diff %.3g\n", 49 + c, argmax(q), argmax(w), maximum(abs.(q .- w))); end
L = fetch(S, S.G.names[:lse], P - 50); lse = L[1, :]; ce = [lse[c] - sub[ids[50 + c + 1] + 1, c] for c in 1:P-51]
@printf("host CE over rows 50..94 from the kernel's logits: %.5f; kernel's lse edge mean %.5f\n", sum(ce) / length(ce), sum(lse) / length(lse))
