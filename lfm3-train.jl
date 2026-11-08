# lfm3-train.jl: what lfm3-data wrote, windows of it through lfm3.jl's gradient and step.
#   julia --project=env/cuda -t 8 lfm3-train.jl --train DATADIR --model DIR --out OUTDIR [--epochs E --lr L --lrv L --lre L --beta B --acc A --windows W --seed S]
@isdefined(load) || include(joinpath(@__DIR__, "lfm3.jl"))
# ---- training: what lfm3-data wrote ----
struct Data; tok::Vector{Int32}; tr::Vector{NTuple{5,UInt32}}; ex::Vector{NTuple{4,UInt32}}; end
function Data(dir)
    tok = reinterpret(Int32, read(joinpath(dir, "tokens.bin"))) |> collect
    t = reinterpret(UInt32, read(joinpath(dir, "trans.bin"))); e = reinterpret(UInt32, read(joinpath(dir, "examples.bin")))
    Data(tok, [(t[6i+1], t[6i+2], t[6i+3], t[6i+4], t[6i+5]) for i in 0:length(t)÷6-1], [(e[4i+1], e[4i+2], e[4i+3], e[4i+4]) for i in 0:length(e)÷4-1])
end
tfirst(D::Data, t) = Int(D.tr[t+1][1]) | Int(D.tr[t+1][2]) << 32
function windows(D::Data, held::Bool)                     
    W = Tuple{Int,Int,Int,Vector{Int},Vector{Int}}[]; i = 1; ne = length(D.ex)
    while i <= ne
        t, w0, _, _ = D.ex[i]; last = i
        while last + 1 <= ne && D.ex[last+1][1] == t && D.ex[last+1][4] <= w0 + NPMAX[]; last += 1; end
        if (D.tr[t+1][5] != 0) == held
            N = Int(D.ex[last][4] - w0); pos = Int[]; tgt = Int[]; f = tfirst(D, Int(t))
            for j in i:last
                _, _, rs, re = D.ex[j]
                for p in Int(rs)-1:Int(re)-2; push!(pos, p - Int(w0)); push!(tgt, D.tok[f+p+2]); end
            end
            sh = max(0, N - NPMAX[])                          # a window holds what the pool holds: a longer one starts later (its loss rows before that go)
            if sh > 0; k = pos .>= sh; pos = pos[k] .- sh; tgt = tgt[k]; N -= sh; end
            isempty(pos) || push!(W, (Int(t), f + Int(w0) + sh, N, pos, tgt))
        end
        i = last + 1
    end
    W
end
# one window: the loss, its gradient when wanted
function window!(S::Sys, D::Data, w; grad = true, f2 = 1.0, iacc = 1.0)
    t, f0, N, pos, tgt = w
    rows = Matrix{Int32}(undef, 4, N); for p in 0:N-1; rows[:, p+1] = Int32[0, p, D.tok[f0+p+1], -1]; end
    for (p, g) in zip(pos, tgt); rows[4, p+1] = g; end
    request!(S, rows, reshape(Int32[-1, N], 2, 1), Int32.(pos), ones(Float32, length(pos)), grad ? 6 : 2; f2, iacc)
    L = fetch(S, S.G.names[:lse], length(pos)); sum(L[2, :]) / length(pos)
end
step!(S::Sys) = request!(S, reshape(Int32[0, 0, 0, -1], 4, 1), reshape(Int32[-1, 1], 2, 1), Int32[0], Float32[1], 8)
function heldout(S::Sys, D::Data, W)
    t0 = time(); tot = 0.0; n = 0
    for w in W; tot += window!(S, D, w; grad = false) * length(w[4]); n += length(w[4]); end
    @printf("held out: %.4f over %d reply tokens in %d windows (%.0f s)\n", tot / n, n, length(W), time() - t0); flush(stdout)
    tot / n
end
# every trained tensor as a Float32 vector (momenta: name * ".momentum")
function trained(S::Sys; only = nothing)                
    G = S.G; byname = Dict{String,Vector{Float32}}(); want(nm) = only === nothing || (endswith(nm, ".master") ? nm[1:end-7] : nm) in only
    for (nm, p, nb) in S.host; want(nm) && (byname[nm] = copy(unsafe_wrap(Array, Ptr{Float32}(p), nb ÷ 4))); end
    for ((m, which), b) in S.vram; nm = G.matname[m] * (which == 4 ? "" : ".momentum"); want(nm) && (byname[nm] = dn!(Vector{Float32}(undef, length(b) ÷ 4), b, 0)); end
    for C in S.cards
        dev!(C.i + 1)
        for (v, nm) in enumerate(G.vecname)
            C.vec[VF*(v-1)+1] == 0 && continue; n = veclen(nm); off = Int(C.vec[VF*(v-1)+1] - addr(C.bufs[2]))
            byname[nm] = dn!(Vector{Float32}(undef, n), C.bufs[2], off); S.train && (byname[nm*".momentum"] = dn!(Vector{Float32}(undef, n), C.bufs[3], off))
        end
    end
    for k in collect(keys(byname)); endswith(k, ".master") && (byname[k[1:end-7]] = byname[k]; delete!(byname, k)); end
    byname
end
function restore!(S::Sys, byname)                          # the inverse
    G = S.G
    for (nm, p, nb) in S.host; k = endswith(nm, ".master") ? nm[1:end-7] : nm; haskey(byname, k) && copyto!(unsafe_wrap(Array, Ptr{Float32}(p), nb ÷ 4), byname[k]); end
    for ((m, which), b) in S.vram; nm = G.matname[m] * (which == 4 ? "" : ".momentum"); haskey(byname, nm) && up!(b, 0, byname[nm]); end
    for C in S.cards
        dev!(C.i + 1)
        for (v, nm) in enumerate(G.vecname)
            C.vec[VF*(v-1)+1] == 0 && continue; off = Int(C.vec[VF*(v-1)+1] - addr(C.bufs[2]))
            haskey(byname, nm) && up!(C.bufs[2], off, byname[nm]); haskey(byname, nm * ".momentum") && up!(C.bufs[3], off, byname[nm*".momentum"])
        end
    end
end
function bf16(a::AbstractVector{Float32})
    o = Vector{UInt16}(undef, length(a))
    Threads.@threads for i in eachindex(a); u = reinterpret(UInt32, a[i]); o[i] = ((u + 0x00007fff + ((u >> 16) & 0x00000001)) >> 16) % UInt16; end
    o
end
function export_bf16(S::Sys, out)
    mkpath(out)
    for f in readdir(S.dir); (f == "model.safetensors" || endswith(f, ".bak-nobos") || startswith(f, "ref.")) || cp(joinpath(S.dir, f), joinpath(out, f); force = true); end
    byname = trained(S); names = sort(collect(keys(S.g.t))); hdr = IOBuffer(); off = 0; print(hdr, "{")
    for (i, nm) in enumerate(names)
        shape = S.g.t[nm][1]; n = prod(shape); length(byname[nm]) == n || error("$nm: $(length(byname[nm])) values, shape $shape")
        print(hdr, i == 1 ? "" : ",", "\"", nm, "\":{\"dtype\":\"BF16\",\"shape\":[", join(shape, ","), "],\"data_offsets\":[", off, ",", off + 2n, "]}"); off += 2n
    end
    print(hdr, "}"); h = take!(hdr); pad = (8 - length(h) % 8) % 8; append!(h, fill(UInt8(' '), pad))
    open(joinpath(out, "model.safetensors"), "w") do io
        write(io, UInt64(length(h))); write(io, h); for nm in names; write(io, bf16(byname[nm])); end
    end
    println("wrote $out/model.safetensors"); flush(stdout)
end
# a checkpoint: masters, momenta, where the epoch stood; written whole, renamed
function ckpt_write(S::Sys, path, ep, i, best)
    t0 = time(); tmp = path * ".tmp"; byname = trained(S); names = sort(collect(keys(byname)))
    open(tmp, "w") do io
        write(io, Int64(ep), Int64(i), Float64(best), Int64(length(names)))
        for nm in names; write(io, Int64(length(nm))); write(io, nm); write(io, Int64(length(byname[nm]))); write(io, byname[nm]); end
    end
    mv(tmp, path; force = true); @printf("checkpoint: epoch %d window %d (%.0f s)\n", ep, i, time() - t0); flush(stdout)
end
function ckpt_read(S::Sys, path)
    byname = Dict{String,Vector{Float32}}(); ep = 0; i = 0; best = 0.0
    open(path) do io
        ep = read(io, Int64); i = read(io, Int64); best = read(io, Float64); n = read(io, Int64)
        for _ in 1:n; l = read(io, Int64); nm = String(read(io, l)); m = read(io, Int64); byname[nm] = read!(io, Vector{Float32}(undef, m)); end
    end
    restore!(S, byname); step0!(S)
    @printf("resumed: epoch %d window %d, best held out %.4f\n", ep, i, best); flush(stdout)
    Int(ep), Int(i), best
end
step0!(S::Sys) = (for n in 1:S.G.nn; S.G.node[NF*(n-1)+N_KIND+1] == STEP && (S.G.nodef[4(n-1)+1] = 0); end; step!(S))   # W from the masters: a step of 0
# the gradient check: one window (momenta with f2 = 0, acc = 1)
# a window's length: what the pool holds, and what a window runs in `seconds` by the time measured here (a gradient pass over two short windows, a fixed cost and a cost a position)
function calibrate!(S::Sys, D::Data, seconds)
    cap = NPMAX[]; ns = Int[]; ts = Float64[]
    for n in (64, 192)
        NPMAX[] = n; w = windows(D, false)[1]; dt = 0.0
        for _ in 1:2; t0 = time(); window!(S, D, w; f2 = 1.0, iacc = 0.0); dt = time() - t0; end   # (the second run: warm; iacc 0 leaves the momentum as it was)
        push!(ns, w[3]); push!(ts, dt)
    end
    b = (ts[2] - ts[1]) / (ns[2] - ns[1]); a = ts[1] - b * ns[1]
    n = clamp(floor(Int, (0.9 * seconds - a) / b), 64, cap)
    for _ in 1:3                                           # the window of that length, timed: longer ones cost more a position (attention), so shrink to the budget
        NPMAX[] = n; w = windows(D, false)[1]; dt = 0.0
        for _ in 1:2; t0 = time(); window!(S, D, w; f2 = 1.0, iacc = 0.0); dt = time() - t0; end
        dt <= 0.95 * seconds || n <= 64 || (n = max(64, floor(Int, n * 0.9 * seconds / dt))); dt <= 0.95 * seconds && break
    end
    NPMAX[] = n
    @printf("%.2f s + %.1f ms a position: windows of at most %d positions (the pool holds %d)\n", a, 1e3 * b, NPMAX[], cap); NPMAX[]
end
function train(args)
    arg(k, d) = (i = findfirst(==(k), args); i === nothing ? d : args[i+1])
    dir = arg("--model", ""); data = arg("--train", ""); out = arg("--out", ""); epochs = parse(Int, arg("--epochs", "2"))
    lr = parse(Float64, arg("--lr", "0.002")); lrv = parse(Float64, arg("--lrv", "0.0002")); lre = parse(Float64, arg("--lre", "0.002")); beta = parse(Float64, arg("--beta", "0.95"))
    acc = parse(Int, arg("--acc", "8")); nwin = parse(Int, arg("--windows", "0")); seed = parse(Int, arg("--seed", "1")); reserve = parse(Float64, arg("--reserve", "3.5e9"))
    out == "" || mkpath(out); (dir == "" || data == "") && error("usage: lfm3.jl --train DATADIR --model DIR --out OUTDIR [--epochs E --lr L --lrv L --lre L --beta B --acc A --windows W --check PREFIX N --resume]")
    S = load(dir; train = true, nslot = 0, lr, lrv, lre, reserve); D = Data(data); calibrate!(S, D, parse(Float64, arg("--seconds", "10")))
    Wt = windows(D, false); Wh = windows(D, true)
    nwin > 0 && (Wt = Wt[1:min(nwin, end)]; Wh = Wh[1:min(cld(nwin, 3), end)])
    @printf("%d training windows, %d held out; %d windows a step\n", length(Wt), length(Wh), acc); flush(stdout)
    ck = joinpath(out, "checkpoint.bin"); ep0 = 1; i0 = 0
    if "--resume" in args && isfile(ck); ep0, i0, best = ckpt_read(S, ck) else; best = heldout(S, D, Wh); end
    rng = Random.MersenneTwister(seed); for _ in 1:ep0-1; Random.shuffle(rng, 1:length(Wt)); end
    for ep in ep0:epochs
        order = Random.shuffle(rng, 1:length(Wt)); t0 = time(); tot = 0.0; n = 0
        for (i, wi) in enumerate(order)
            ep == ep0 && i <= i0 && continue
            w = Wt[wi]; t1 = time(); loss = window!(S, D, w; f2 = (i - 1) % acc == 0 ? beta : 1.0, iacc = 1 / acc)
            tot += loss * length(w[4]); n += length(w[4])
            @printf("epoch %d window %d/%d: N %d, %d reply tokens, loss %.4f (%.1f s)\n", ep, i, length(order), w[3], length(w[4]), loss, time() - t1); flush(stdout)
            (i % acc == 0 || i == length(order)) && (t2 = time(); step!(S); @printf("step (%.1f s)\n", time() - t2))
            i % (acc * 32) == 0 && ckpt_write(S, ck, ep, i, best)
        end
        @printf("epoch %d: train loss %.4f over %d reply tokens (%.0f min)\n", ep, tot / n, n, (time() - t0) / 60)
        h = heldout(S, D, Wh)
        h < best ? (best = h; export_bf16(S, out)) : println("not the best ($best): no export")
        ckpt_write(S, ck, ep, length(order), best); flush(stdout)
    end
end
(abspath(PROGRAM_FILE) == @__FILE__) && train(ARGS)
