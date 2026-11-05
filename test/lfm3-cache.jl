# lfm3-cache.jl: against a running lfm3-1.7b.jl's socket, on the engine's own protocol. A resumed prompt must give the tokens
# of a cold run, past the conv and sliding rings' lengths; top_k 1 must give the argmax; a seed must repeat and another differ.
#   julia test/lfm3-cache.jl SOCK REF.ids
using Sockets
sock, ids = ARGS; I = Int.(reinterpret(Int32, read(ids)))
function chat(prompt, n; snap = 0, sample = "")
    c = connect(sock); isempty(sample) || println(c, "sample ", sample); snap > 0 && println(c, "snap ", snap)
    println(c, "chat ", n, " ", join(prompt, ' ')); flush(c); t = Int[]; d = ""
    while !eof(c)
        l = readline(c); startswith(l, "tok") && push!(t, parse(Int, l[5:end])); startswith(l, "done") && (d = l; break)
    end
    close(c); t, split(d)
end
ok = true; check(name, c) = (println(c ? "PASS " : "FAIL ", name); global ok &= c)
flush_slots() = for j in 1:4; chat(I[400+20j:440+20j], 1; snap = 30); end
for (n1, s1, n2) in ((400, 390, 600), (300, 299, 310), (40, 30, 500))
    flush_slots()
    chat(I[1:n1], 1; snap = s1)                            # leaves a snapshot at s1
    a, d = chat(I[1:n2], 48)                               # resumes from it
    check("prompt $n1 snapshot $s1, then $n2 resumes $s1", parse(Int, d[7]) == s1)
    flush_slots()                                          # other conversations take the slots: the snapshot is gone
    b, e = chat(I[1:n2], 48)
    check("  that run is cold", parse(Int, e[7]) == 0)
    check("  resumed tokens == cold tokens ($(length(a)))", a == b)
end
g, _ = chat(I[1:100], 48)
t1, _ = chat(I[1:100], 48; sample = "7 0.7 1 1.0")
check("top_k 1 == the argmax", t1 == g)
s1, _ = chat(I[1:100], 48; sample = "7 0.8 40 0.95"); s2, _ = chat(I[1:100], 48; sample = "7 0.8 40 0.95"); s3, _ = chat(I[1:100], 48; sample = "8 0.8 40 0.95")
check("the same seed repeats", s1 == s2)
check("another seed differs", s1 != s3)
check("a seed differs from the argmax", s1 != g)
check("tokens in the vocabulary", all(t -> 0 <= t < 128000, [s1; s3]))
t2, _ = chat(I[1:100], 48; sample = "7 0.7 40 0.0001"); check("top_p near 0 == the argmax", t2 == g)
# the draws against the softmax of the reference's logits (a third argument, REF.logits.bf16): temperature and top-k
if length(ARGS) >= 3
    using Mmap
    N = 16; T = 1.5; K = 6; D = 300
    r = Mmap.mmap(ARGS[3], Matrix{UInt16}, (128000, 600)); v = [Float64(reinterpret(Float32, UInt32(x) << 16)) for x in view(r, :, N)]
    o = sortperm(v; rev = true)[1:K]; p = exp.(v[o] ./ T); p ./= sum(p); n = Dict(i - 1 => 0 for i in o)
    for s in 1:D
        t, _ = chat(I[1:N], 1; sample = "$(1000 + s) $T $K 1.0"); n[t[1]] = get(n, t[1], 0) + 1
    end
    chi = sum((n[o[j]-1] - D * p[j])^2 / (D * p[j]) for j in 1:K)
    println("expected ", round.(D .* p; digits = 1), "\ndrawn    ", [n[i-1] for i in o], "\nchi-square ", round(chi; digits = 2), " on $(K - 1) degrees of freedom")
    check("the draws stay in the top $K", sum(values(n)) == sum(n[i-1] for i in o))
    check("the draws fit the softmax (chi-square under 20.5, p about 0.001)", chi < 20.5)
end
ok || exit(1)
