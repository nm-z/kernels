# serve.jl: lfm3.jl as an engine of mux (codex/PROTOCOL.md): token ids in, token ids out; mux renders the chat template, tokenizes and parses.
#   julia --project=env/cuda -t 8 lfm3-serve/serve.jl --model DIR [--cap N]      (mux starts it with MUX_SOCK and MUX_SERVICE)
# The slot keeps the last run's cache: the next prompt runs from where it differs. Sampling is the model's own (its GGUF: temp 0.7, top-k 20, top-p 0.95).
@isdefined(load) || include(joinpath(@__DIR__, "..", "lfm3.jl"))
include(joinpath(@__DIR__, "..", "codex", "julia", "Mux.jl")); using .Mux
const TEMP, TOPK, TOPP = 0.7, 20, 0.95
function slotrun(S::Sys, toks, p0)                           # rows p0.. of the slot's chat; the last row's logits
    n = length(toks); rows = Int32[r == 1 ? 0 : r == 2 ? p0 + i - 1 : r == 3 ? toks[i] : -1 for r in 1:4, i in 1:n]
    request!(S, rows, reshape(Int32[0, p0 + n], 2, 1), Int32[n - 1], ones(Float32, 1), 1); fetch(S, S.G.names[:lg], 1)[:, 1]
end
function pick(l, rng)
    ix = partialsortperm(l, 1:TOPK; rev = true); p = exp.((l[ix] .- l[ix[1]]) ./ TEMP); c = cumsum(p); k = something(findfirst(>=(TOPP * c[end]), c), TOPK)
    ix[findfirst(>=(rand(rng) * c[k]), c[1:k])] - 1
end
function run!(S::Sys, r::Mux.Run, held::Vector{Int}, cap::Int)
    ids = r.ids; P = length(ids); (0 < P < cap) || return Mux.fail!(r, "$P positions: the engine holds $cap")
    t0 = time(); busy = 0.0; fin = Threads.Atomic{Bool}(false)
    beat = Threads.@spawn while !fin[]; Mux.stats!(r); sleep(0.25); end   # (stats at least every 500 ms, a prefill request is longer)
    idle() = Mux.idle!(r, 100 * max(0.0, 1 - busy / max(time() - t0, 1e-9)))                 # the share of the run the host is outside a request: sampling, copies, the socket
    p0 = 0; while p0 < min(P - 1, length(held)) && held[p0+1] == ids[p0+1]; p0 += 1; end   # the cache's prefix; the last token always runs
    length(held) - p0 > 100 && (p0 = 0); resize!(held, p0); l = Float32[]
    for a in p0+1:64:P                                       # (64 rows a request: the rings)
        Mux.cancelled(r) && break
        b = min(P, a + 63); tb = time(); l = slotrun(S, ids[a:b], a - 1); busy += time() - tb; append!(held, ids[a:b]); Mux.prefill!(r, b); idle()
    end
    tp = time(); gen = 0; stopped = false; t1 = tp; rng = Random.Xoshiro(hash(r.id))
    while !isempty(l) && gen < r.max_new && !Mux.cancelled(r)
        t = pick(l, rng); gen += 1; gen == 1 && (t1 = time()); Mux.tokens!(r, [t])
        t in r.stop && (stopped = true; break); gen >= r.max_new && break
        tb = time(); l = slotrun(S, [t], P + gen - 1); busy += time() - tb; push!(held, t); idle()
    end
    t2 = time(); fin[] = true; wait(beat)
    pt = (P - p0) / max(min(t1, t2) - t0, 1e-9); dt = gen > 1 ? (gen - 1) / max(t2 - t1, 1e-9) : 0.0
    Mux.done!(r; prefill_tps = pt, decode_tps = dt, idle_pct = 100 * max(0.0, 1 - busy / max(t2 - t0, 1e-9)), stopped)
    @printf("run %s: %d prompt (%d cached), %d out, prefill %.2f tok/s, decode %.2f tok/s%s\n", r.id, P, p0, gen, pt, dt, Mux.cancelled(r) ? " (cancelled)" : ""); flush(stdout)
end
function main(dir; cap = 4096)
    S = load(dir; nslot = 1, cap); held = Int[]; lk = ReentrantLock()
    println("registering with mux: ", cap, " positions"); flush(stdout)
    Mux.serve(positions = cap, max_new = 2048, slots = 1) do r; lock(lk) do; run!(S, r, held, cap); end; end
end
if abspath(PROGRAM_FILE) == @__FILE__
    a = ARGS; arg(k, d) = (i = findfirst(==(k), a); i === nothing ? d : a[i+1])
    main(arg("--model", ""); cap = parse(Int, arg("--cap", "4096")))
end
