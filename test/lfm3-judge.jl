# lfm3-judge.jl: what the served model says as the user, on held-out contexts, beside what the user said. Each example's
# context (its turns up to and including "<|im_start|>user\n") goes to the engine's gen; the reply is cut at <|im_end|>.
#   julia --project=env/amd test/lfm3-judge.jl DIR DATADIR SOCK [N examples] [tokens]
include(joinpath(@__DIR__, "..", "lfm3-1.7b.jl"))
using Sockets
dir, data, sock = ARGS[1:3]; nex = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 8; ntok = length(ARGS) > 4 ? parse(Int, ARGS[5]) : 48
D = Data(data); vocab = vocabulary(dir)
held = [i for (i, e) in enumerate(D.ex) if D.tr[e[1]+1][5] != 0 && e[3] - e[2] <= 1024 && e[4] - e[3] >= 6]
bytr = Dict{Int,Vector{Int}}(); for i in held; push!(get!(bytr, Int(D.ex[i][1]), Int[]), i); end   # one example a transcript, its middle one
mids = sort([v[(length(v) + 1) ÷ 2] for v in values(bytr)])
picks = mids[round.(Int, range(1, length(mids); length = min(nex, length(mids))))]
cut(t) = (i = findfirst(==(IM_END), t); i === nothing ? t : t[1:i-1])
const IM_END = 124900
for i in picks
    t, cs, rs, re = Int.(D.ex[i]); f = tfirst(D, t)
    ctx = D.tok[f+cs+1:f+rs]; actual = D.tok[f+rs+1:f+re]
    c = connect(sock); println(c, "gen ", ntok, " ", join(ctx, ' ')); flush(c)
    out = Int[]
    while !eof(c)
        l = readline(c)
        startswith(l, "ids: ") && (out = parse.(Int, split(l[6:end])))
        startswith(l, "idle time") && break
    end
    close(c)
    gen = out
    println("== example $i (transcript $t, $(rs - cs) context tokens); the last assistant turn's tail:")
    println("   ", repr(last(detok(vocab, ctx[max(1, end - 60):end]), 240)))
    println("   user said:  ", repr(detok(vocab, cut(actual))))
    println("   model says: ", repr(detok(vocab, cut(gen))))
end
