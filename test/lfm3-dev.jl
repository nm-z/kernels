# lfm3-dev.jl: what the tests read off a loaded lfm3.jl system: a host progress watch (LFM3_WATCH=s), the per-node-kind profile (DBG 64),
# Include after lfm3.jl.
@isdefined(Data) || include(joinpath(@__DIR__, "..", "lfm3-train.jl"))
const WATCH = Ref(parse(Float64, get(ENV, "LFM3_WATCH", "0")))
function await!(S::Sys)                                   # (LFM3_WATCH=s: progress)
    (WATCH[] > 0 && CUDAON) || return; p = Ptr{UInt32}(S.cntH); last = -1; t = time()
    ev = [(dev!(C.i + 1); e = CUDA.CuEvent(); CUDA.record(e); e) for C in S.cards]
    say(m) = (h = Int(unsafe_load(p, NFLAG - 2)); c = Int(unsafe_load(p, NFLAG - 1)); n = clamp(Int(unsafe_load(p, NFLAG)), 1, S.G.nn);
              @printf("%s: %d / %d tasks, node %d kind %d l %d\n", m, c, h >> 4, n, S.G.node[NF*(n-1)+N_KIND+1], S.G.node[NF*(n-1)+N_L+1]); flush(stdout); c)
    try
        while !all(CUDA.isdone, ev); sleep(0.05); time() - t >= WATCH[] || continue; t = time(); c = say("watch"); c == last && println("  (still)"); last = c; end
    catch; say("fault after"); rethrow(); end
end
function prof(S::Sys, C::Card)                            # (debug 64: cycles per node kind)
    nn = S.G.nn; st = dn!(Vector{UInt32}(undef, 4(nn + 1)), C.cnt, 4(NFLAG - 4(nn + 1))); up!(C.cnt, 4(NFLAG - 4(nn + 1)), zeros(UInt32, 4(nn + 1)))
    by = Dict{Int,Vector{Float64}}(); for n in 1:nn; v = get!(by, S.G.node[NF*(n-1)+N_KIND+1], zeros(3)); v .+= st[4n+1:4n+3]; end
    @printf("card %d (%d blocks): derive %.1f Mcyc\n", C.i, C.nblk, st[1] * 256 / 1e6)
    for (k, v) in sort(collect(by); by = x -> -x[2][2]); @printf("  kind %2d: wait %8.1f  run %8.1f Mcyc  %6d tasks\n", k, v[1] * 256 / 1e6, v[2] * 256 / 1e6, v[3]); end
    for n in sort(1:nn; by = n -> -Int(st[4n+2]))[1:6]; @printf("  node %4d kind %2d l %2d: wait %8.1f  run %8.1f Mcyc  %6d tasks\n", n, S.G.node[NF*(n-1)+N_KIND+1], S.G.node[NF*(n-1)+N_L+1], st[4n+1] * 256 / 1e6, st[4n+2] * 256 / 1e6, st[4n+3]); end
end
# one window's gradients (the momenta of the CHECKN tensors) to PREFIX.NAME.f32
const CHECKN = ["model.embed_tokens.weight", "model.embedding_norm.weight", ["model.layers.$l.$t.weight" for l in (23, 22, 0) for t in ("operator_norm", "post_operator_norm", "ffn_norm", "post_ffn_norm", "feed_forward.w1", "feed_forward.w3", "feed_forward.w2")]...,
                ["model.layers.$l.conv.$t.weight" for l in (23, 0) for t in ("in_proj", "out_proj", "conv")]..., ["model.layers.22.self_attn.$t.weight" for t in ("q_proj", "k_proj", "v_proj", "gate_proj", "o_proj")]...]
function gradcheck(S::Sys, D::Data, prefix, N)
    W = windows(D, false); w = W[findfirst(w -> any(p -> p + 1 < N, w[4]), W)]; t, f0, _, pos, tgt = w
    keep = findall(p -> p + 1 < N, pos); pos = pos[keep]; tgt = tgt[keep]; isempty(pos) && (pos = [N - 2]; tgt = [D.tok[f0+N]])
    write(prefix * ".ids", Int32[D.tok[f0+p+1] for p in 0:N-1]); write(prefix * ".pos", Int32.(pos))
    t0 = time(); loss = window!(S, D, (t, f0, N, pos, tgt); f2 = 0.0, iacc = 1.0)
    @printf("check: N %d, %d loss positions, loss %.5f (%.1f s)\n", N, length(pos), loss, time() - t0); write(prefix * ".loss", Float32(loss))
    byname = trained(S; only = Set(nm * ".momentum" for nm in CHECKN)); for nm in CHECKN; write(prefix * "." * nm * ".f32", byname[nm*".momentum"]); end
    println("wrote $prefix.*.f32"); flush(stdout)
end
