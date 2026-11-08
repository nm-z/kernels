# lfm3-prof.jl: where a forward's cycles go, per node kind (debug 64)
#   julia --project=env/cuda -t 8 test/lfm3-prof.jl DIR N
include(joinpath(@__DIR__, "..", "lfm3.jl")); include(joinpath(@__DIR__, "lfm3-dev.jl"))
S = load(ARGS[1]; nslot = 0, reserve = 0.6e9, hostgb = 2.0); N = parse(Int, ARGS[2])
ids = Int.(reinterpret(Int32, read(joinpath(ARGS[1], "ref.ids"))))[1:N]
DBG[] = 64; logits(S, ids); for C in S.cards; prof(S, C); end
t0 = time(); logits(S, ids); dt = time() - t0; @printf("%d positions: %.2f s (%.1f tok/s)\n", N, dt, N / dt); for C in S.cards; prof(S, C); end
