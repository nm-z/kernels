# lfm3-upd.jl: one matrix's Muon step in isolation: the momentum set to a known matrix, then the orthogonalized step
# against a Float64 Newton-Schulz on the host, the master against master - lr step, and the packed W and Wt against
# a repack of the master.
#   julia --project=env/amd -t 12 test/lfm3-upd.jl DIR
include(joinpath(@__DIR__, "..", "lfm3-1.7b.jl"))
using LinearAlgebra, Random
dir = ARGS[1]
S = load_train(dir); B = TB(); KTR[] = k_tr!(ROCBackend(), Int(WGS))
function ns(G)                                             # rows x K; the same polynomial either way round
    X = G / (norm(G) + 1e-7)
    for _ in 1:5
        if size(X, 1) <= size(X, 2)
            A = X * X'; Bm = NSB * A + NSC * A * A; X = NSA * X + Bm * X
        else
            A = X' * X; Bm = NSB * A + NSC * A * A; X = NSA * X + X * Bm
        end
    end
    X
end
for (l, j) in ((2, 2), (0, 1), (0, 5), (2, 4))                     # k_proj (512 x 2048), conv in (6144 x 2048), w2 (2048 x 7680), gate (16 x 2048)
    tw = S.L[l+1].M[j]; R, K = tw.rows, tw.K
    rng = MersenneTwister(l * 10 + j); G = randn(rng, Float32, R, K) .* 1f-3   # [r][k] row-major: Julia's transpose
    m = unsafe_wrap(Array, tw.moh, R * K); m .= vec(permutedims(G)); m0 = copy(unsafe_wrap(Array, tw.mh, R * K))
    lr = 0.01; muon!(B, tw, lr); AMDGPU.synchronize()
    X = permutedims(reshape(Array(B.X)[1:R*K], K, R)); X2 = permutedims(reshape(Array(B.X2)[1:R*K], K, R))
    O = ns(Float64.(G))
    err = min(norm(X - O) / norm(O), norm(X2 - O) / norm(O))        # (whichever buffer the last swap left it in)
    Xo = norm(X - O) <= norm(X2 - O) ? X : X2
    sv = svdvals(Float64.(Xo));
    m1 = unsafe_wrap(Array, tw.mh, R * K); f = lr * sqrt(max(1.0, R / K))
    dm = norm(m1 - (m0 - f * vec(permutedims(Xo)))) / norm(f * vec(permutedims(Xo)))
    mm = reshape(m1, K, R)'                                        # [r][k]
    Wn = Array(tw.W.W)
    Wr = reinterpret(UInt8, repack(bf16(vec(permutedims(Float32.(mm)))), R, K)) |> collect
    println(tw.name, ": NS rel err ", round(err; sigdigits = 3), ", singular values ", round(minimum(sv); digits = 3), " .. ", round(maximum(sv); digits = 3),
            ", master step rel err ", round(dm; sigdigits = 3), ", W bytes differ ", count(Wn .!= Wr), " of ", length(Wr))
    m .= 0
end
