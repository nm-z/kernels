include(joinpath(@__DIR__, "..", "lfm3.jl"))
S = load(expanduser("~/models/lfm3-1.7b-base"); nslot = 0, reserve = 0.6e9, hostgb = 2.0)
ids = Int.(reinterpret(Int32, read(expanduser("~/models/lfm3-1.7b-base/ref.ids"))))
kl(S, ids, expanduser("~/models/lfm3-1.7b-base/ref.logits.bf16"), parse(Int, ARGS[1]))
