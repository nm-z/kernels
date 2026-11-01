# q27-cancel.jl: on sentry, against the running engine. A 3000-token prompt streams progress ("prog" lines); the
# client then goes (half-close, as flashd's cancel_on_eof does), and within 15 s the engine stops: the four dies are
# idle and back at auto. Under a minute.
#   julia test/q27-cancel.jl /data/models/q27.sock
using Sockets
sock = get(ARGS, 1, "/data/models/q27.sock")
dies = filter(d -> isfile(joinpath(d, "gpu_busy_percent")), ["/sys/class/drm/$c/device" for c in readdir("/sys/class/drm") if occursin(r"^card\d+$", c)])
busy() = [parse(Int, strip(read(joinpath(d, "gpu_busy_percent"), String))) for d in dies]
level() = [strip(read(joinpath(d, "power_dpm_force_performance_level"), String)) for d in dies]
fails = String[]
check(ok, what) = (println(ok ? "ok   " : "FAIL ", what); ok || push!(fails, what))

check(length(dies) == 4, "four dies ($(length(dies)))")
check(all(==("auto"), level()), "dies at auto before the request: $(level())")
c = connect(sock); t0 = time()
println(c, "timings 1"); println(c, "cancel_on_eof 1"); println(c, "chat 400 ", join(fill(1000, 3000), ' '))
progs = String[]; line = ""
while !eof(c) && time() - t0 < 40
    global line = readline(c)
    startswith(line, "prog ") && (push!(progs, line); break)
end
check(!isempty(progs) && progs[1] == "prog 1024 3000 0", "first progress line after the first prefill forward: $(repr(line))")
check(all(==("high"), level()), "dies forced high during the request: $(level())")
closewrite(c); tc = time()
while !eof(c) && time() - tc < 20; readline(c); end           # the engine closes when its kernel has stopped
ts = time() - tc; close(c)
check(ts < 15, "the request stopped $(round(ts, digits = 1)) s after the client went")
sleep(1)
check(all(<(5), busy()), "dies idle after the cancel: $(busy()) %")
check(all(==("auto"), level()), "dies back at auto after the cancel: $(level())")
println(isempty(fails) ? "PASS" : "FAIL: $(length(fails))", " ($(round(time() - t0, digits = 1)) s)")
exit(isempty(fails) ? 0 : 1)
