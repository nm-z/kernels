# serve.jl: lfm3.jl as an engine of mux (codex/PROTOCOL.md): token ids in, token ids out; mux renders the chat template, tokenizes and parses.
#   julia --project=env/cuda -t 8 lfm3-serve/serve.jl --model DIR [--cap N] [--slots N]      (mux starts it with MUX_SOCK and MUX_SERVICE)
# The slot keeps the last run's cache: the next prompt runs from where it differs. Sampling is the model's own (lfm3.jl: temp 0.7, top-k 20, top-p 0.95): the first token of a prompt on the host,
# then the card samples by itself: the chats decoding together run in chains of launches (chain.jl) that the host only watches, so the card does not wait for it.
@isdefined(load) || include(joinpath(@__DIR__, "..", "lfm3.jl"))
include(joinpath(@__DIR__, "chain.jl"))
include(joinpath(@__DIR__, "..", "codex", "julia", "Mux.jl")); using .Mux
struct Job; slot::Int; toks::Vector{Int}; p0::Int; out::Channel{Vector{Float32}}; end
mutable struct Lane; slot::Int; pos::Int; tok::Int; out::Channel{Int}; done::Bool; gone::Base.Event; end   # a chat decoding: the token at pos, the tokens out; done: the chat is over, gone: the batcher dropped it
const SAMP = Ref(2); const INBOX = Channel{Any}(Inf)                                 # the jobs (a prompt's rows) and the lanes (a chat that starts decoding) waiting for the card
const CHUNK = Ref(1); const ROWS = Ref(typemax(Int))                                                      # the rows one request has been seen not to hold (none yet: every job waiting goes together)
# the rows a slot can take in one request: what its rings hold past what a row reads back (lfm3.jl sizes a ring to the next power of two of reach + BURST)
chunk(S::Sys) = minimum(nextpow(2, r + BURST) - r for r in (S.G.edge[EF*(e-1)+E_REACH+1] for e in 1:S.G.ne) if r > 0)
function prefill!(S::Sys, jobs)                                 # the jobs waiting run as the segments of one request (the weights are read once for all of them)
    take = Job[]; nrow = 0
    while !isempty(jobs) && length(take) < NSEG && (isempty(take) || nrow + length(jobs[1].toks) < ROWS[]); j = popfirst!(jobs); push!(take, j); nrow += length(j.toks); end
    sort!(take; by = j -> j.slot); NS = length(take); rows = Matrix{Int32}(undef, 4, nrow); segs = Matrix{Int32}(undef, 2, NS); lr = Int32[]; c = 0
    for (s, j) in enumerate(take)
        n = length(j.toks); segs[:, s] = Int32[j.slot, j.p0 + n]
        for i in 1:n; c += 1; rows[:, c] = Int32[s - 1, j.p0 + i - 1, j.toks[i], -1]; end
        push!(lr, c - 1)
    end
    try
        request!(S, rows, segs, lr, ones(Float32, NS), 1); L = fetch(S, S.G.names[:lg], NS)
        for (s, j) in enumerate(take); Base.put!(j.out, L[:, s]); end   # (lfm3.jl's own put! and fetch shadow Base's)
    catch e
        if length(take) > 1 && e isa ErrorException && occursin(r"status [12],", e.msg)   # the plan did not fit (nothing ran): fewer rows a request from now on, these jobs again
            ROWS[] = nrow; prepend!(jobs, take)
        else; for j in take; close(j.out, e); end; end
    end
end
function decode!(S::Sys, lanes, cap)                            # up to 32 steps of every chat decoding, the tokens handed out as they come; a prompt, a chat that ends or one that starts stops the chain
    for l in lanes; cap - 1 - l.pos < 1 && (l.done = true; close(l.out); notify(l.gone)); end
    filter!(l -> !l.done, lanes); isempty(lanes) && return
    K = min(KMAX, minimum(cap - 1 - l.pos for l in lanes))      # (as many steps as the chain's buffers hold: an arrival aborts what has not started)
    ch = chain!(S, [l.slot for l in lanes], [l.pos for l in lanes], [l.tok for l in lanes], K; seed = rand(UInt32), samp = SAMP[]); k = 1; aborted = false
    while k <= K
        if chainready(ch, k)
            for (i, t) in enumerate(chaintoks(ch, k)); l = lanes[i]; l.done || (l.pos += 1; l.tok = t; Base.put!(l.out, t)); end
            k += 1
        elseif aborted
            chaindone() && break; sleep(0.0005)
        else
            (isready(INBOX) || any(l -> l.done, lanes)) && (chainabort!(S); aborted = true); sleep(0.0005)
        end
    end
    chainend!(ch)
end
function batcher(S::Sys, cap)                                   # the one task that owns the card
    dev!(1); lanes = Lane[]; jobs = Job[]; sink(x) = x isa Job ? push!(jobs, x) : push!(lanes, x)
    while true
        try
            isempty(jobs) && isempty(lanes) && sink(take!(INBOX))
            while isready(INBOX); sink(take!(INBOX)); end
            for l in lanes; l.done && (close(l.out); notify(l.gone)); end; filter!(l -> !l.done, lanes)
            if !isempty(jobs); prefill!(S, jobs)
            elseif !isempty(lanes); sleep(0.002); isready(INBOX) || decode!(S, lanes, cap)   # (a prompt's next rows may be on their way: a moment's grace)
            end
        catch e
            @error "batcher" exception = (e, catch_backtrace()); flush(stderr); exit(1)             # (mux starts the engine again)
        end
    end
end
function slotrun(slot, toks, p0)                                # rows p0.. of a slot's chat; the last row's logits
    j = Job(slot, toks, p0, Channel{Vector{Float32}}(1)); Base.put!(INBOX, j); take!(j.out)
end
function topk(l, k)                                             # the k largest, best first, in one pass (a threshold of the kept: almost every logit is skipped at once)
    v = fill(-Inf32, k); ix = zeros(Int, k); m = -Inf32; mi = 1
    @inbounds for i in eachindex(l)
        x = l[i]; x > m && (v[mi] = x; ix[mi] = i; mi = argmin(v); m = v[mi])
    end
    ix[sortperm(v; rev = true)]
end
function pick(l, rng)                                           # (the sampler of lfm3.jl's SAMPLE node, for the first token of a prompt)
    ix = topk(l, STOPK); p = exp.((l[ix] .- l[ix[1]]) ./ STEMP); c = cumsum(p); k = something(findfirst(>=(STOPP * c[end]), c), STOPK)
    ix[findfirst(>=(rand(rng) * c[k]), c[1:k])] - 1
end
function run!(slot::Int, r::Mux.Run, held::Vector{Int}, cap::Int)
    ids = r.ids; P = length(ids); (0 < P < cap) || return Mux.fail!(r, "$P positions: the engine holds $cap")
    t0 = time(); busy = 0.0; fin = Threads.Atomic{Bool}(false)
    idle() = Mux.idle!(r, 100 * max(0.0, 1 - busy / max(time() - t0, 1e-9)))                 # the share of the run no request of it is in flight: the host between its prompt's chunks, the socket
    beat = Threads.@spawn while !fin[]; Mux.stats!(r); idle(); sleep(0.25); end             # (stats at least every 500 ms, a prefill request is longer)
    p0 = 0; while p0 < min(P - 1, length(held)) && held[p0+1] == ids[p0+1]; p0 += 1; end   # the cache's prefix; the last token always runs
    length(held) - p0 > 100 && (p0 = 0); resize!(held, p0); l = Float32[]
    for a in p0+1:CHUNK[]:P                                  # (a request's rows of one slot: what the rings hold)
        Mux.cancelled(r) && break
        b = min(P, a + CHUNK[] - 1); tb = time(); l = slotrun(slot, ids[a:b], a - 1); busy += time() - tb; append!(held, ids[a:b]); Mux.prefill!(r, b)
    end
    gen = 0; stopped = false; t1 = time(); lane = nothing
    if !isempty(l) && r.max_new > 0 && !Mux.cancelled(r)
        t = pick(l, Random.Xoshiro(hash(r.id))); gen = 1; t1 = time(); Mux.tokens!(r, [t]); stopped = t in r.stop
        if !stopped && gen < r.max_new
            lane = Lane(slot, P, t, Channel{Int}(Inf), false, Base.Event()); Base.put!(INBOX, lane); tb = time()
            while !stopped && gen < r.max_new && !Mux.cancelled(r)
                n = try; take!(lane.out); catch; break; end                                # (closed: the slot is full)
                push!(held, t); t = n; gen += 1; Mux.tokens!(r, [t]); stopped = t in r.stop
            end
            lane.done = true; busy += time() - tb
        end
    end
    t2 = time(); fin[] = true; wait(beat)
    pt = (P - p0) / max(min(t1, t2) - t0, 1e-9); dt = gen > 1 ? (gen - 1) / max(t2 - t1, 1e-9) : 0.0
    Mux.done!(r; prefill_tps = pt, decode_tps = dt, idle_pct = 100 * max(0.0, 1 - busy / max(t2 - t0, 1e-9)), stopped)
    @printf("run %s: %d prompt (%d cached), %d out, prefill %.2f tok/s, decode %.2f tok/s%s\n", r.id, P, p0, gen, pt, dt, Mux.cancelled(r) ? " (cancelled)" : ""); flush(stdout)
    lane === nothing || wait(lane.gone)                      # (the card is done with the slot before another chat takes it)
end
function main(dir; cap = 4096, nslot = 4)
    S = load(dir; nslot, cap); CHUNK[] = chunk(S); helds =[Int[] for _ in 1:nslot]; free = Set(0:nslot-1); lk = ReentrantLock(); cv = Threads.Condition(lk)
    Threads.@spawn batcher(S, cap)
    println("registering with mux: ", nslot, " slots of ", cap, " positions"); flush(stdout)
    Mux.serve(positions = cap, max_new = 2048, slots = nslot) do r
        pre(s) = (h = helds[s+1]; n = 0; while n < min(length(h), length(r.ids) - 1) && h[n+1] == r.ids[n+1]; n += 1; end; n)
        slot = lock(lk) do; while isempty(free); wait(cv); end; s = argmax(pre, sort!(collect(free))); delete!(free, s); s; end   # (the free slot whose cache shares the longest prefix with the prompt)
        try; run!(slot, r, helds[slot+1], cap); finally; lock(lk) do; push!(free, slot); notify(cv); end; end
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    a = ARGS; arg(k, d) = (i = findfirst(==(k), a); i === nothing ? d : a[i+1])
    main(arg("--model", ""); cap = parse(Int, arg("--cap", "4096")), nslot = parse(Int, arg("--slots", "4")))
end
