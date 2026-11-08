# lfm3-edges.jl: the forward's edges of a few positions against the same math on the CPU, layer 0 (conv) and layer 2 (attention)
#   julia --project=env/amd -t 8 test/lfm3-edges.jl DIR N
include(joinpath(@__DIR__, "..", "lfm3.jl"))
dir = ARGS[1]; N = parse(Int, ARGS[2])
S = load(dir; nslot = 0, reserve = 0.6e9, hostgb = 2.0); g = S.g; G = S.G
ids = Int.(reinterpret(Int32, read(joinpath(dir, "ref.ids"))))[1:N]
DBG[] = 1; Lg = logits(S, ids)
T(name) = Float32[bf32(u) for u in tensor(g, name)]
M(name) = permutedims(reshape(T(name), g.t[name][1][2], g.t[name][1][1]))   # [rows][K] row-major on disk -> rows x K
Emb = reshape(T("model.embed_tokens.weight"), NE, NV)
x = Matrix{Float32}(undef, NE, N); for p in 1:N; x[:, p] = Emb[:, ids[p]+1]; end
rms(a) = 1.0f0 ./ sqrt.(vec(sum(a .^ 2; dims = 1)) ./ size(a, 1) .+ EPS)
fr = rftab()
function cmp(name, e, ref)                                 # ref: width x N
    got = fetch(S, e, N); d = maximum(abs.(got .- ref)); m = maximum(abs.(ref))
    @printf("%-8s max|diff| %.4g of max|ref| %.4g%s\n", name, d, m, d > 0.02 * m + 1e-3 ? "   <-- off" : "")
end
cmp("x0", G.names[:x0], x)
for l in 0:NL-1
    L(s) = G.names[Symbol(l, s)]; pre = "model.layers.$l."
    rs1 = rms(x); cmp("rs1", L(:rs1), reshape(rs1, 1, N)); xn = x .* rs1' .* T(pre * "operator_norm.weight")
    if isattn(l)
        qkv = vcat(M(pre * "self_attn.q_proj.weight") * xn, M(pre * "self_attn.k_proj.weight") * xn, M(pre * "self_attn.v_proj.weight") * xn, M(pre * "self_attn.gate_proj.weight") * xn)
        cmp("qkv", L(:qkv), qkv)
        q = reshape(qkv[1:NE, :], HD, NH, N); k = reshape(qkv[NE+1:NE+NKV*HD, :], HD, NKV, N); v = reshape(qkv[NE+NKV*HD+1:NE+2NKV*HD, :], HD, NKV, N); gt = qkv[NE+2NKV*HD+1:end, :]
        rsq = [1.0f0 / sqrt(sum(q[:, j, p] .^ 2) / HD + EPS) for j in 1:NH, p in 1:N]; rsk = [1.0f0 / sqrt(sum(k[:, j, p] .^ 2) / HD + EPS) for j in 1:NKV, p in 1:N]
        cmp("rsq", L(:rsq), rsq); cmp("rsk", L(:rsk), rsk)
        wq = T(pre * "self_attn.q_norm.weight"); wk = T(pre * "self_attn.k_norm.weight"); tbl = isfull(l) ? 0 : 1
        function rot(a, p)                                 # a: HD, position p (0-based)
            o = similar(a)
            for i in 0:63; an = Float32(p) * fr[tbl*64+i+1]; c = cos(an); s = sin(an); o[i+1] = a[i+1] * c - a[i+65] * s; o[i+65] = a[i+65] * c + a[i+1] * s; end
            o
        end
        qr = similar(q); kr = similar(k)
        for p in 1:N, j in 1:NH; qr[:, j, p] = rot(q[:, j, p] .* wq .* rsq[j, p], p - 1); end
        for p in 1:N, j in 1:NKV; kr[:, j, p] = rot(k[:, j, p] .* wk .* rsk[j, p], p - 1); end
        cmp("qr", L(:qr), reshape(qr, NE, N)); cmp("kr", L(:kr), reshape(kr, NKV * HD, N)); cmp("v", L(:v), reshape(v, NKV * HD, N))
        ao = zeros(Float32, HD, NH, N); lse = zeros(Float32, NH, N)
        for p in 1:N, j in 1:NH
            gk = (j - 1) ÷ 4 + 1; lo = isfull(l) ? 1 : max(1, p - WIN + 1); sc = [SCALE * dot(qr[:, j, p], kr[:, gk, m]) for m in lo:p]
            mx = maximum(sc); e = exp.(sc .- mx); lse[j, p] = mx + log(sum(e)); pr = e ./ sum(e)
            ao[:, j, p] = sum(pr[i] .* v[:, gk, lo+i-1] for i in 1:length(pr)) .* (1.0f0 / (1.0f0 + exp(-gt[j, p])))
        end
        cmp("lse", L(:lse), lse); cmp("ao", L(:ao), reshape(ao, NE, N))
        o = M(pre * "self_attn.o_proj.weight") * reshape(ao, NE, N)
    else
        bcx = M(pre * "conv.in_proj.weight") * xn; cmp("bcx", L(:bcx), bcx)
        h = bcx[1:NE, :] .* bcx[2NE+1:3NE, :]; taps = reshape(T(pre * "conv.conv.weight"), 7, NE)
        cv = zeros(Float32, NE, N); for p in 1:N, t in 0:6; m = p - 6 + t; m >= 1 && (cv[:, p] .+= taps[t+1, :] .* h[:, m]); end
        y = bcx[NE+1:2NE, :] .* cv; cmp("y", L(:y), y); cmp("cv", L(:cv), cv)
        o = M(pre * "conv.out_proj.weight") * y
    end
    cmp("o", L(:o), o); rso = rms(o); cmp("rso", L(:rso), reshape(rso, 1, N))
    x1 = x .+ o .* rso' .* T(pre * "post_operator_norm.weight"); cmp("x1", L(:x1), x1)
    rsf = rms(x1); fn = x1 .* rsf' .* T(pre * "ffn_norm.weight"); u1 = M(pre * "feed_forward.w1.weight") * fn; u3 = M(pre * "feed_forward.w3.weight") * fn
    cmp("u", L(:u), vcat(u1, u3)); s = u1 ./ (1 .+ exp.(-u1)) .* u3; cmp("s", L(:s), s)
    f = M(pre * "feed_forward.w2.weight") * s; cmp("f", L(:f), f); rsff = rms(f)
    global x = x1 .+ f .* rsff' .* T(pre * "post_ffn_norm.weight"); cmp("x$(l+1)", G.names[Symbol(l + 1, :x0)], x)
end
