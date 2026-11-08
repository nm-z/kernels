# lfm3-upd2.jl: one step of lfm3.jl against a Float64 Newton-Schulz on the host: the momentum of a few matrices set to known values
# (the others zero), then the masters' change against lr f ns(G), and the BF16 weights against the masters.
#   julia --project=env/amd -t 8 test/lfm3-upd2.jl DIR
include(joinpath(@__DIR__, "..", "lfm3.jl")); include(joinpath(@__DIR__, "lfm3-dev.jl"))
using LinearAlgebra, Random
S = load(ARGS[1]; train = true, nslot = 0, reserve = 2e9, hostgb = 6.0, lr = 0.01, lrv = 0.001, lre = 0.01); G = S.G
function ns(X)
    X = X / (norm(X) + 1e-7)
    for _ in 1:5
        if size(X, 1) <= size(X, 2); A = X * X'; B = NSB * A + NSC * A * A; X = NSA * X + B * X
        else; A = X' * X; B = NSB * A + NSC * A * A; X = NSA * X + X * B; end
    end
    X
end
names = ["model.layers.2.self_attn.k_proj.weight", "model.layers.0.conv.in_proj.weight", "model.layers.0.feed_forward.w2.weight", "model.layers.2.self_attn.gate_proj.weight"]; before = trained(S; only = Set(names))
Gs = Dict(nm => randn(MersenneTwister(i), Float32, matdims(nm)...) .* 1f-3 for (i, nm) in enumerate(names))
mo = Dict(nm * ".momentum" => vec(permutedims(Gs[nm])) for nm in names); restore!(S, mo)
DBG[] = 64; t0 = time(); step!(S); @printf("step: %.1f s\n", time() - t0); for C in S.cards; prof(S, C); end; after = trained(S; only = Set(names))
for nm in names
    R, K = matdims(nm); d = (after[nm] .- before[nm]); X = ns(Float64.(Gs[nm])); ref = -0.01 * sqrt(max(1.0, R / K)) .* vec(permutedims(X))
    @printf("%-48s step rel %.3g  cos %.5f  |ref| %.3g\n", nm, norm(d .- ref) / norm(ref), dot(d, ref) / (norm(d) * norm(ref)), norm(ref))
end
