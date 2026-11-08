# AiMux.jl against a fake ai-mux (a Unix socket in this process): register, a run with progress, tokens, stats
# and done, a cancel, and the JSON round trip. Under a few seconds.
include(joinpath(@__DIR__, "..", "AiMux.jl")); using .AiMux, Sockets, Test
sock = tempname() * ".sock"; srv = listen(sock)
ENV["AI_MUX_SOCK"] = sock; ENV["AI_MUX_SERVICE"] = "fake"
engine = Threads.@spawn AiMux.serve(positions = 1000, max_new = 50, slots = 1) do r
    for p in (256, 512, length(r.ids)); AiMux.idle!(r, 12.5); AiMux.prefill!(r, p); sleep(0.3); end
    for t in 1:10
        AiMux.cancelled(r) && break
        AiMux.tokens!(r, [100 + t]); sleep(0.06)
    end
    AiMux.done!(r; prefill_tps = 123.4, decode_tps = 45.6, idle_pct = 7.8, stopped = false)
end
c = accept(srv)
reg = AiMux.jparse(readline(c))
@test reg["op"] == "register" && reg["service"] == "fake" && reg["kind"] == "engine" && reg["positions"] == 1000 && reg["max_new"] == 50
println(c, """{"op":"registered","ctx":900}""")
println(c, """{"op":"run","id":"r1","ids":[$(join(1:600, ','))],"max_new":10,"stop":[7],"images":[]}""")
lines = Dict{String,Any}[]
while true
    m = AiMux.jparse(readline(c)); push!(lines, m)
    m["op"] == "done" && break
end
ops = [m["op"] for m in lines]
@test count(==("prefill"), ops) >= 2 && "stats" in ops
@test [m["done"] for m in lines if m["op"] == "prefill"][end] == 600
toks = reduce(vcat, [m["ids"] for m in lines if m["op"] == "tokens"])
@test toks == collect(101:110)
d = lines[end]
@test d["prompt"] == 600 && d["generated"] == 10 && d["prefill_tps"] ≈ 123.4 && d["idle_pct"] ≈ 7.8 && d["stopped"] == false
@test all(m -> m["decode_tps"] >= 0 && m["idle_pct"] == 12.5, filter(m -> m["op"] == "stats", lines))
# a cancel stops the run early
println(c, """{"op":"run","id":"r2","ids":[1,2,3],"max_new":10,"stop":[],"images":[]}""")
sleep(1.2); println(c, """{"op":"cancel","id":"r2"}""")
ncancel = let n = 0
    while true
        m = AiMux.jparse(readline(c)); m["op"] == "tokens" && (n += length(m["ids"])); m["op"] == "done" && break
    end
    n
end
@test ncancel < 10
@test AiMux.jparse(AiMux.jobj(:a => "x\"\n\\", :b => [1, 2], :c => 1.5, :d => nothing, :e => true)) ==
      Dict("a" => "x\"\n\\", "b" => [1, 2], "c" => 1.5, "d" => nothing, "e" => true)
close(c); wait(engine); rm(sock)
