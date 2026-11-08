# AiMux.jl: a Julia engine's side of the ai-mux protocol (../PROTOCOL.md). The engine registers with
# the ai-mux on this host and runs what it is sent; this module does the socket, the JSON lines and the
# live numbers, so an engine only turns token ids into token ids:
#
#   include("AiMux.jl"); using .AiMux
#   AiMux.serve(positions = 262144, max_new = 8192, slots = 1) do r     # r::AiMux.Run, in its own task
#       ... run r.ids (stop at r.stop, at most r.max_new new tokens) ...
#       AiMux.prefill!(r, done)            # prompt positions run so far (rate-limited here)
#       AiMux.tokens!(r, ids)              # new token ids, as they are made
#       AiMux.idle!(r, pct)                # the idle % of the run so far, measured by the engine
#       AiMux.cancelled(r) && ...          # the client went: stop early
#       AiMux.done!(r; prefill_tps, decode_tps, idle_pct, stopped)   # the engine's measured values
#   end
#
# The live prefill and decode tok/s ai-mux streams while a run goes are measured here from when the
# engine reports progress and tokens; the final ones are the engine's own. serve returns when ai-mux
# closes the connection (it stops the service's process then anyway).
module AiMux
using Sockets, Printf

# ---- JSON: the flat objects of the protocol (numbers, strings, bools, null, arrays, objects) ----
jstr(s) = (io = IOBuffer(); print(io, '"'); for c in s
    c == '"' ? print(io, "\\\"") : c == '\\' ? print(io, "\\\\") : c == '\n' ? print(io, "\\n") :
    c < ' ' ? @printf(io, "\\u%04x", UInt32(c)) : print(io, c)
end; print(io, '"'); String(take!(io)))
jval(v::AbstractString) = jstr(v)
jval(v::Bool) = v ? "true" : "false"
jval(v::Integer) = string(v)
jval(v::AbstractFloat) = isfinite(v) ? @sprintf("%.6g", v) : "null"
jval(::Nothing) = "null"
jval(v::AbstractVector) = "[" * join(jval.(v), ",") * "]"
jobj(ps::Pair...) = "{" * join((jstr(String(k)) * ":" * jval(v) for (k, v) in ps), ",") * "}"

function jparse(s::AbstractString)
    i = Ref(firstindex(s))
    ws() = while i[] <= ncodeunits(s) && s[i[]] in (' ', '\t', '\n', '\r'); i[] = nextind(s, i[]); end
    function val()
        ws(); c = s[i[]]
        if c == '{'
            d = Dict{String,Any}(); i[] += 1; ws()
            s[i[]] == '}' && (i[] += 1; return d)
            while true
                ws(); k = val(); ws(); s[i[]] == ':' || error("json: ':' expected"); i[] += 1
                d[k] = val(); ws()
                s[i[]] == ',' ? (i[] += 1) : s[i[]] == '}' ? (i[] += 1; return d) : error("json: ',' or '}' expected")
            end
        elseif c == '['
            a = Any[]; i[] += 1; ws()
            s[i[]] == ']' && (i[] += 1; return a)
            while true
                push!(a, val()); ws()
                s[i[]] == ',' ? (i[] += 1) : s[i[]] == ']' ? (i[] += 1; return a) : error("json: ',' or ']' expected")
            end
        elseif c == '"'
            io = IOBuffer(); i[] += 1
            while (c = s[i[]]) != '"'
                if c == '\\'
                    i[] += 1; e = s[i[]]
                    if e == 'u'
                        print(io, Char(parse(UInt16, s[i[]+1:i[]+4]; base = 16))); i[] += 4
                    else
                        print(io, e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e == 'b' ? '\b' : e == 'f' ? '\f' : e)
                    end
                else
                    print(io, c)
                end
                i[] = nextind(s, i[])
            end
            i[] += 1; return String(take!(io))
        elseif startswith(SubString(s, i[]), "true"); i[] += 4; return true
        elseif startswith(SubString(s, i[]), "false"); i[] += 5; return false
        elseif startswith(SubString(s, i[]), "null"); i[] += 4; return nothing
        else
            j = i[]; while i[] <= ncodeunits(s) && s[i[]] in "+-0123456789.eE"; i[] += 1; end
            t = s[j:i[]-1]; return occursin(r"[.eE]", t) ? parse(Float64, t) : parse(Int, t)
        end
    end
    val()
end

# ---- a run ----
mutable struct Run
    id::String
    ids::Vector{Int}               # the prompt
    max_new::Int
    stop::Vector{Int}
    images::Vector{Any}            # [{img, at}]
    conn::Any
    t0::Float64                    # when the run arrived
    tpre::Float64                  # when the last prompt position ran (0: not yet)
    tfirst::Float64                # when the first token came
    prefilled::Int
    ntok::Int
    tsent::Float64                 # when stats last went out
    stopflag::Threads.Atomic{Bool}
    finished::Bool
    idle::Float64                  # the engine's live idle % of the run so far (NaN: not measured yet)
end
struct Conn
    io::IO
    lk::ReentrantLock
end
send(c::Conn, line) = lock(c.lk) do
    write(c.io, line, '\n'); flush(c.io)
end
send(r::Run, line) = send(r.conn, line)

"Prompt positions run so far (`done` of the run's `length(r.ids)`)."
function prefill!(r::Run, done::Integer)
    now = time(); total = length(r.ids); done = min(done, total)
    done <= r.prefilled && return
    r.prefilled = done
    done == total && (r.tpre = now)
    if done == total || now - r.tsent >= 0.25
        send(r, jobj(:op => "prefill", :id => r.id, :done => done, :total => total)); stats!(r, now)
    end
end
"New token ids, as the engine makes them."
function tokens!(r::Run, ids)
    isempty(ids) && return
    now = time()
    r.prefilled < length(r.ids) && prefill!(r, length(r.ids))
    r.ntok == 0 && (r.tfirst = now)
    r.ntok += length(ids)
    send(r, jobj(:op => "tokens", :id => r.id, :ids => collect(Int, ids)))
    now - r.tsent >= 0.25 && stats!(r, now)
end
function stats!(r::Run, now = time())
    tp = r.tpre > 0 ? r.tpre : now
    ptps = r.prefilled / max(tp - r.t0, 1e-6)
    dtps = r.ntok > 1 ? (r.ntok - 1) / max(now - r.tfirst, 1e-6) : 0.0
    send(r, jobj(:op => "stats", :id => r.id, :prefill_tps => ptps, :decode_tps => dtps, :idle_pct => r.idle)); r.tsent = now
end
"Whether the client went (stop early, then call done!)."
cancelled(r::Run) = r.stopflag[]
"The engine's idle % of the run so far (goes out with the next stats line)."
idle!(r::Run, pct::Real) = (r.idle = Float64(pct); nothing)
"The run's end, with the engine's measured values."
function done!(r::Run; prefill_tps, decode_tps, idle_pct, stopped::Bool)
    r.finished && return
    r.prefilled < length(r.ids) && prefill!(r, length(r.ids))
    stats!(r)
    send(r, jobj(:op => "done", :id => r.id, :prompt => length(r.ids), :generated => r.ntok,
        :prefill_tps => Float64(prefill_tps), :decode_tps => Float64(decode_tps), :idle_pct => Float64(idle_pct), :stopped => stopped))
    r.finished = true
end
fail!(r::Run, msg) = (r.finished || send(r, jobj(:op => "error", :id => r.id, :message => msg)); r.finished = true)

"""
    serve(f; positions, max_new, slots = 1, vision = false, encode = nothing)

Connect to `ENV["AI_MUX_SOCK"]` as service `ENV["AI_MUX_SERVICE"]`, register, and run `f(r::Run)` for
each run (at most `slots` at once). `encode(img, bytes) -> positions` handles image encoding when
`vision`. Returns ai-mux's configured context window when the connection closes.
"""
function serve(f; positions::Integer, max_new::Integer, slots::Integer = 1, vision::Bool = false, encode = nothing)
    sock = ENV["AI_MUX_SOCK"]; ep = ENV["AI_MUX_SERVICE"]
    c = Conn(connect(sock), ReentrantLock())
    send(c, jobj(:op => "register", :service => ep, :kind => "engine", :positions => positions, :max_new => max_new, :slots => slots, :vision => vision))
    ctx = 0; runs = Dict{String,Run}(); free = Base.Semaphore(slots)
    while !eof(c.io)
        m = jparse(readline(c.io)); op = m["op"]
        if op == "registered"
            ctx = m["ctx"]; @info "ai-mux: registered as $ep, context $ctx"
        elseif op == "run"
            r = Run(m["id"], Int.(m["ids"]), m["max_new"], Int.(m["stop"]), get(m, "images", Any[]), c,
                    time(), 0.0, 0.0, 0, 0, 0.0, Threads.Atomic{Bool}(false), false, NaN)
            runs[r.id] = r
            Threads.@spawn begin
                Base.acquire(free)
                try
                    f(r)
                    r.finished || fail!(r, "the engine ended the run without done")
                catch e
                    fail!(r, sprint(showerror, e))
                finally
                    Base.release(free); delete!(runs, r.id)
                end
            end
        elseif op == "cancel"
            r = get(runs, m["id"], nothing); r === nothing || (r.stopflag[] = true)
        elseif op == "image"
            vision && encode !== nothing || error("ai-mux sent an image to an engine without vision")
            n = encode(m["img"], base64decode_(m["bytes"]))
            send(c, jobj(:op => "image", :img => m["img"], :tokens => n))
        end
    end
    ctx
end
function base64decode_(s)
    v(c) = c in 'A':'Z' ? c - 'A' : c in 'a':'z' ? c - 'a' + 26 : c in '0':'9' ? c - '0' + 52 : c in "+-" ? 62 : 63
    out = UInt8[]; acc = UInt32(0); bits = 0
    for c in s
        c in "=\n\r " && continue
        acc = (acc << 6) | UInt32(v(c)); bits += 6
        bits >= 8 && (bits -= 8; push!(out, UInt8((acc >> bits) & 0xff)))
    end
    out
end

end
