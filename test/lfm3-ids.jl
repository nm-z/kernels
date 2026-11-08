# lfm3-ids.jl: the logits of the kernel at PREFIX.pos for PREFIX.ids, written as PREFIX.logits.f32 (PL x NV, row-major by position)
#   julia --project=env/cuda -t 8 test/lfm3-ids.jl DIR PREFIX
include(joinpath(@__DIR__, "..", "lfm3.jl"))
S = load(ARGS[1]; nslot = 0, reserve = 0.6e9, hostgb = 2.0); prefix = ARGS[2]
ids = Int.(reinterpret(Int32, read(prefix * ".ids"))); pos = reinterpret(Int32, read(prefix * ".pos"))
Lg = logits(S, ids; lrows = Int32.(pos)); write(prefix * ".logits.f32", Float32.(vec(Lg)))
ce = [log(sum(exp.(Lg[:, c] .- maximum(Lg[:, c])))) + maximum(Lg[:, c]) - Lg[ids[pos[c] + 2] + 1, c] for c in 1:length(pos)]
@printf("kernel: %d positions, mean CE %.5f (first rows %s)\n", length(pos), sum(ce) / length(ce), join([@sprintf("%.3f", x) for x in ce[1:min(6, end)]], " "))
