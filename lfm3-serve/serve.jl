# serve.jl: lfm3.jl as an engine of mux (codex/PROTOCOL.md): token ids in, token ids out; mux renders the chat template, tokenizes and parses.
#   julia --project=env/cuda -t 8 lfm3-serve/serve.jl --model DIR [--cap N]      (mux starts it with MUX_SOCK and MUX_SERVICE)
# The slot keeps the last run's cache: the next prompt runs from where it differs. Sampling is the model's own (its GGUF: temp 0.7, top-k 20, top-p 0.95).
@isdefined(load) || include(joinpath(@__DIR__, "..", "lfm3.jl"))
include(joinpath(@__DIR__, "..", "codex", "julia", "Mux.jl")); using .Mux
const TEMP, TOPK, TOPP = 0.7, 20, 0.95
struct Job; slot::Int; toks::Vector{Int}; p0::Int; out::Channel{Vector{Float32}}; end
const JOBS = Channel{Job}(Inf); const ACTIVE = Threads.Atomic{Int}(0)   # the jobs waiting; the chats running
function batcher(S::Sys, maxrows)                            # the one task that owns the card: the jobs waiting run as the segments of one request (the weights are read once for all of them)
    while true
        jobs = [take!(JOBS)]; nrow = length(jobs[1].toks); t0 = time()
        while nrow < maxrows && (isready(JOBS) || (length(jobs) < ACTIVE[] && time() - t0 < 0.03))   # (every running chat has a job in flight but for the few ms its host samples)
            isready(JOBS) ? (j = take!(JOBS); push!(jobs, j); nrow += length(j.toks)) : sleep(0.0005)
        end
        sort!(jobs; by = j -> j.slot); NS = length(jobs); rows = Matrix{Int32}(undef, 4, nrow); segs = Matrix{Int32}(undef, 2, NS); lr = Int32[]; c = 0
        for (s, j) in enumerate(jobs)
            n = length(j.toks); segs[:, s] = Int32[j.slot, j.p0 + n]
            for i in 1:n; c += 1; rows[:, c] = Int32[s - 1, j.p0 + i - 1, j.toks[i], -1]; end
            push!(lr, c - 1)
        end
        try
            request!(S, rows, segs, lr, ones(Float32, NS), 1); L = fetch(S, S.G.names[:lg], NS)
            for (s, j) in enumerate(jobs); Base.put!(j.out, L[:, s]); end   # (lfm3.jl's own put! and fetch shadow Base's)
        catch e
            for j in jobs; close(j.out, e); end
        end
    end
end
function slotrun(slot, toks, p0)                             # rows p0.. of a slot's chat; the last row's logits
    j = Job(slot, toks, p0, Channel{Vector{Float32}}(1)); Base.put!(JOBS, j); take!(j.out)
end
function topk(l, k)                                          # the k largest, best first, in one pass (a threshold of the kept: almost every logit is skipped at once)
    v = fill(-Inf32, k); ix = zeros(Int, k); m = -Inf32; mi = 1
    @inbounds for i in eachindex(l)
        x = l[i]; x > m && (v[mi] = x; ix[mi] = i; mi = argmin(v); m = v[mi])
    end
    ix[sortperm(v; rev = true)]
end
function pick(l, rng)
    ix = topk(l, TOPK); p = exp.((l[ix] .- l[ix[1]]) ./ TEMP); c = cumsum(p); k = something(findfirst(>=(TOPP * c[end]), c), TOPK)
    ix[findfirst(>=(rand(rng) * c[k]), c[1:k])] - 1
end
function run!(slot::Int, r::Mux.Run, held::Vector{Int}, cap::Int)
    ids = r.ids; P = length(ids); (0 < P < cap) || return Mux.fail!(r, "$P positions: the engine holds $cap")
    t0 = time(); busy = 0.0; fin = Threads.Atomic{Bool}(false)
    beat = Threads.@spawn while !fin[]; Mux.stats!(r); sleep(0.25); end   # (stats at least every 500 ms, a prefill request is longer)
    idle() = Mux.idle!(r, 100 * max(0.0, 1 - busy / max(time() - t0, 1e-9)))                 # the share of the run the host is outside a request: sampling, copies, the socket
    p0 = 0; while p0 < min(P - 1, length(held)) && held[p0+1] == ids[p0+1]; p0 += 1; end   # the cache's prefix; the last token always runs
    length(held) - p0 > 100 && (p0 = 0); resize!(held, p0); l = Float32[]
    for a in p0+1:64:P                                       # (64 rows a request: the rings)
        Mux.cancelled(r) && break
        b = min(P, a + 63); tb = time(); l = slotrun(slot, ids[a:b], a - 1); busy += time() - tb; append!(held, ids[a:b]); Mux.prefill!(r, b); idle()
    end
    tp = time(); gen = 0; stopped = false; t1 = tp; rng = Random.Xoshiro(hash(r.id))
    while !isempty(l) && gen < r.max_new && !Mux.cancelled(r)
        t = pick(l, rng); gen += 1; gen == 1 && (t1 = time()); Mux.tokens!(r, [t])
        t in r.stop && (stopped = true; break); gen >= r.max_new && break
        tb = time(); l = slotrun(slot, [t], P + gen - 1); busy += time() - tb; push!(held, t); idle()
    end
    t2 = time(); fin[] = true; wait(beat)
    pt = (P - p0) / max(min(t1, t2) - t0, 1e-9); dt = gen > 1 ? (gen - 1) / max(t2 - t1, 1e-9) : 0.0
    Mux.done!(r; prefill_tps = pt, decode_tps = dt, idle_pct = 100 * max(0.0, 1 - busy / max(t2 - t0, 1e-9)), stopped)
    @printf("run %s: %d prompt (%d cached), %d out, prefill %.2f tok/s, decode %.2f tok/s%s\n", r.id, P, p0, gen, pt, dt, Mux.cancelled(r) ? " (cancelled)" : ""); flush(stdout)
end
function main(dir; cap = 4096, nslot = 4)
    S = load(dir; nslot, cap); helds = [Int[] for _ in 1:nslot]; free = Set(0:nslot-1); lk = ReentrantLock()
    Threads.@spawn batcher(S, 128)
    println("registering with mux: ", nslot, " slots of ", cap, " positions"); flush(stdout)
    Mux.serve(positions = cap, max_new = 2048, slots = nslot) do r
        pre(s) = (h = helds[s+1]; n = 0; while n < min(length(h), length(r.ids) - 1) && h[n+1] == r.ids[n+1]; n += 1; end; n)
        slot = lock(lk) do; s = argmax(pre, sort!(collect(free))); delete!(free, s); s; end   # (the free slot whose cache shares the longest prefix with the prompt)
        Threads.atomic_add!(ACTIVE, 1)
        try; run!(slot, r, helds[slot+1], cap); finally; Threads.atomic_add!(ACTIVE, -1); lock(lk) do; push!(free, slot); end; end
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    a = ARGS; arg(k, d) = (i = findfirst(==(k), a); i === nothing ? d : a[i+1])
    main(arg("--model", ""); cap = parse(Int, arg("--cap", "4096")), nslot = parse(Int, arg("--slots", "4")))
end
