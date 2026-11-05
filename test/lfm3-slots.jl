# lfm3-slots.jl: against a running lfm3-1.7b.jl's socket. Requests running together must give the tokens each gives alone;
# one joining or leaving mid-way must not change the others; each conversation keeps its own prompt cache.
#   julia test/lfm3-slots.jl SOCK REF.ids
using Sockets
sock, ids = ARGS; I = Int.(reinterpret(Int32, read(ids)))
function chat(p, n; snap = 0, sample = "", penalty = "", leave = 0)
    c = connect(sock); isempty(sample) || println(c, "sample ", sample); isempty(penalty) || println(c, "penalty ", penalty); snap > 0 && println(c, "snap ", snap)
    println(c, "chat ", n, " ", join(p, ' ')); flush(c); t = Int[]; d = String[]
    while !eof(c)
        l = readline(c)
        if startswith(l, "tok")
            push!(t, parse(Int, l[5:end])); (leave > 0 && length(t) == leave) && break
        end
        startswith(l, "done") && (d = split(l); break)
        startswith(l, "error") && (println(l); break)
    end
    close(c); t, d
end
ok = true; check(name, c) = (println(c ? "PASS " : "FAIL ", name); global ok &= c)
P = [I[1:100], I[100:400], I[50:70], I[200:500], I[1:30]]; N = 64
solo = [chat(p, N)[1] for p in P]
together(specs) = fetch.([(sleep(d); Threads.@spawn chat(p, n; kw...)) for (d, p, n, kw) in specs])
r = together([(0.0, P[j], N, ()) for j in 1:4])
check("4 at once == each alone", all(r[j][1] == solo[j] for j in 1:4))
r = together([(0.0, P[j], N, ()) for j in 1:3])
check("3 at once == each alone", all(r[j][1] == solo[j] for j in 1:3))
r = together([(0.0, P[j], N, ()) for j in 1:5])
check("5 requests on 4 slots: the fifth waits, all == alone", all(r[j][1] == solo[j] for j in 1:5))
r = together([(0.0, P[1], N, ()), (0.15, P[2], N, ()), (0.3, P[3], N, ())])
check("joining mid-way changes nothing", all(r[j][1] == solo[j] for j in 1:3))
a = Threads.@spawn chat(P[2], N); b = Threads.@spawn chat(P[1], N; leave = 5); sleep(0.05); c = Threads.@spawn chat(P[3], N)
check("one leaving mid-way changes nothing", fetch(a)[1] == solo[2] && fetch(c)[1] == solo[3] && fetch(b)[1] == solo[1][1:5])
s1 = chat(P[1], N; sample = "7 0.8 40 0.95")[1]; p1 = chat(P[3], N; penalty = "1.1")[1]
r = together([(0.0, P[1], N, (; sample = "7 0.8 40 0.95")), (0.0, P[2], N, ()), (0.0, P[3], N, (; penalty = "1.1"))])
check("sampling, greedy and a penalty side by side keep to their own settings", r[1][1] == s1 && r[2][1] == solo[2] && r[3][1] == p1)
# two conversations turn by turn: each resumes from its own snapshot
A1 = I[1:200]; B1 = I[300:450]
chat(A1, 1; snap = 190); chat(B1, 1; snap = 140)
_, da = chat(I[1:260], 1; snap = 250); _, db = chat(I[300:520], 1; snap = 210)
check("conversation A resumed from its snapshot", parse(Int, da[7]) == 190)
check("conversation B resumed from its snapshot", parse(Int, db[7]) == 140)
ra = Threads.@spawn chat(I[1:300], 32); rb = Threads.@spawn chat(I[300:560], 32); xa = fetch(ra); xb = fetch(rb)
check("both resume at once", parse(Int, xa[2][7]) == 250 && parse(Int, xb[2][7]) == 210)
for j in 1:4; chat(I[100+20j:140+20j], 1; snap = 30); end  # (other conversations push these out of the slots)
check("resumed tokens == cold tokens", xa[1] == chat(I[1:300], 32)[1] && xb[1] == chat(I[300:560], 32)[1])
ok || exit(1)
