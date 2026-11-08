# serve.jl: lfm3.jl behind flashd's engine protocol on a unix socket.
#   julia --project=env/cuda -t 8 lfm3-serve/serve.jl --model DIR --serve SOCK
@isdefined(load) || include(joinpath(@__DIR__, "..", "lfm3.jl"))
# ---- serving: flashd's engine protocol (flash-sass kern/flashd.c) on a unix socket; the slot keeps the last chat's cache ----
function slotrun(S::Sys, toks, p0)                           # rows p0.. of the slot's chat; the last logits
    n = length(toks); rows = Int32[r == 1 ? 0 : r == 2 ? p0 + i - 1 : r == 3 ? toks[i] : -1 for r in 1:4, i in 1:n]
    request!(S, rows, reshape(Int32[0, p0 + n], 2, 1), Int32[n - 1], ones(Float32, 1), 1); fetch(S, S.G.names[:lg], 1)[:, 1]
end
function pick(l, temp, topk, topp, rng, seen, pen)
    for t in seen; l[t+1] = l[t+1] > 0 ? l[t+1] / pen : l[t+1] * pen; end
    temp <= 0 && return argmax(l) - 1
    ix = partialsortperm(l, 1:topk; rev = true); p = exp.((l[ix] .- l[ix[1]]) ./ temp); c = cumsum(p); k = something(findfirst(>=(topp * c[end]), c), topk)
    ix[findfirst(>=(rand(rng) * c[k]), c[1:k])] - 1
end
function chat(S::Sys, c, n, ids, stops, (temp, topk, topp), seed, pen, held, cap)
    t0 = time(); P = length(ids); gone = false; (P < cap && P > 0) || (println(c, "error: ", P, " positions, the engine holds ", cap); return)
    @async (try; eof(c); catch; end; gone = true)            # (the client closed its side)
    p0 = 0; while p0 < min(P - 1, length(held)) && held[p0+1] == ids[p0+1]; p0 += 1; end   # the cache's prefix; the last token always runs
    length(held) - p0 > 100 && (p0 = 0); resize!(held, p0); l = Float32[]
    for a in p0+1:64:P                                       # (64 rows a request: the rings)
        b = min(P, a + 63); l = slotrun(S, ids[a:b], a - 1); append!(held, ids[a:b]); println(c, "prog ", b, " ", P, " ", p0); flush(c)
    end
    rng = Random.Xoshiro(seed); seen = pen == 1 ? Int[] : copy(ids); gen = 0; ended = false; t1 = t0; n = clamp(n, 1, cap - P)
    while !gone
        t = pick(l, temp, topk, topp, rng, seen, pen); gen += 1; gen == 1 && (t1 = time()); println(c, "tok ", t); flush(c); push!(seen, t); ended = t in stops
        (gen >= n || ended) && break
        l = slotrun(S, [t], P + gen - 1); push!(held, t)
    end
    println(c, "done ", gen, " ", Int(ended), " ", P, " ", round(1e3 * (time() - t0); digits = 1), " ", round(1e3 * (t1 - t0); digits = 1), " ", p0, " 0"); flush(c)
end
function session(S::Sys, c, lk, held, cap)
    stops = Int[]; smp = (0.0, 32, 1.0); seed = UInt64(1); pen = 1.0
    while !eof(c)
        a = split(readline(c)); isempty(a) && continue
        if a[1] in ("timings", "cancel_on_eof"); println(c, "ok: ", join(a, " ")); flush(c)
        elseif a[1] == "stop"; stops = [t for t in parse.(Int, a[2:end]) if t != 0xffffffff]
        elseif a[1] == "sample"; seed = parse(UInt64, a[2]); smp = (parse(Float64, a[3]), parse(Int, a[4]), parse(Float64, a[5]))
        elseif a[1] == "penalty"; pen = parse(Float64, a[2])
        elseif a[1] == "info"; println(c, "info ", cap, " 0"); flush(c)
        elseif a[1] == "chat"; lock(lk) do; chat(S, c, parse(Int, a[2]), parse.(Int, a[3:end]), stops, smp, seed, pen, held, cap); end; return
        end
    end
end
function serve(dir, sock; cap = 4096)
    S = load(dir; nslot = 1, cap); lk = ReentrantLock(); held = Int[]; ispath(sock) && rm(sock); srv = listen(sock); println("serving on ", sock); flush(stdout)
    while true
        c = accept(srv)
        @async try; session(S, c, lk, held, cap); catch e; try; println(c, "error: ", sprint(showerror, e)); catch; end; finally; close(c); end
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    a = ARGS; serve(a[findfirst(==("--model"), a)+1], a[findfirst(==("--serve"), a)+1])
end
