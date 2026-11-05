# lfm3-desktop.jl: does the desktop on the same card keep running while lfm3-1.7b.jl runs one long dispatch? Polls the
# GPU's rings from debugfs (needs sudo) during a chat request; a ring with work pending whose signaled fence stands still
# for STALL seconds cancels the request (the compute and graphics ring timeouts are 10 s). PASS needs: dispatch longer than
# those timeouts, no cancel, the graphics ring advancing throughout, no ring timeout or reset in the kernel log, the same
# kwin_wayland process.
#   julia test/lfm3-desktop.jl SOCK REF.ids NTOKENS
using Sockets, Printf
const STALL = 3.0
sock, ids, n = ARGS[1], ARGS[2], parse(Int, ARGS[3])
kwin() = strip(read(`pgrep -o -x kwin_wayland`, String))
function fenceinfo()
    for d in readlines(`sudo sh -c 'ls -d /sys/kernel/debug/dri/*/ | sort'`)
        s = read(`sudo cat $(d)amdgpu_fence_info`, String)
        occursin("gfx_0.0.0", s) || continue
        r = Dict{String,Tuple{UInt64,UInt64}}()
        for m in eachmatch(r"--- ring \d+ \(([^)]+)\) ---\nLast signaled fence\s+0x([0-9a-f]+)\nLast emitted\s+0x([0-9a-f]+)", s)
            r[m[1]] = (parse(UInt64, m[2], base = 16), parse(UInt64, m[3], base = 16))
        end
        return r
    end
    error("no amdgpu_fence_info with a gfx ring")
end
k0 = kwin(); since = read(`date '+%Y-%m-%d %H:%M:%S'`, String) |> strip
b0 = fenceinfo()["gfx_0.0.0"][1]; sleep(5); base = (fenceinfo()["gfx_0.0.0"][1] - b0) / 5    # the desktop's rate without the kernel
r0 = fenceinfo(); rings = [k for k in keys(r0) if startswith(k, "gfx") || startswith(k, "comp")]
c = connect(sock); ii = Int.(reinterpret(Int32, read(ids)))[1:64]
println(c, "chat ", n, " ", join(ii, ' ')); flush(c)
toks = Ref(0); done = Ref(false); line = Ref("")
reader = @async try
    while !eof(c)
        l = readline(c); startswith(l, "tok") && (toks[] += 1)
        startswith(l, "done") && (line[] = l; done[] = true; break)
    end
catch
    done[] = true
end
last = Dict(k => (r0[k][1], time()) for k in rings); worst = Dict(k => 0.0 for k in rings); t0 = time(); cancelled = false; samples = 0
while !done[] && time() - t0 < 58
    r = fenceinfo(); now = time(); global samples += 1
    for k in rings
        s, e = r[k]; a, ta = last[k]
        if s != a || s == e
            last[k] = (s, now)
        else
            worst[k] = max(worst[k], now - ta)
            if now - ta > STALL && !cancelled
                global cancelled = true; println("STALL on $k: cancelling"); close(c)
            end
        end
    end
    sleep(0.2)
end
wall = time() - t0; isopen(c) && close(c)
r1 = fenceinfo(); sleep(1)
# (the rings say nothing about the screens: after a reset the rings ran while two outputs' page flips stayed stuck)
log = read(pipeline(ignorestatus(`journalctl -k --since $since`), ignorestatus(`rg -i 'ring .*timeout|GPU reset|GPU Hang|flip_done|commit wait'`)), String)
ebusy = read(pipeline(ignorestatus(`journalctl --user --since $since`), ignorestatus(`rg -c 'Atomic commit failed'`)), String) |> strip
isempty(ebusy) || (log *= "kwin: $ebusy failed atomic commits\n")
gfx = r1["gfx_0.0.0"][1] - r0["gfx_0.0.0"][1]
@printf("wall %.1f s, %d tokens, %d polls, %s\n", wall, toks[], samples, line[])
@printf("gfx_0.0.0 fences signaled during the run: %d (%.1f a second; %.1f a second before it)\n", gfx, gfx / wall, base)
for k in sort(rings); worst[k] > 0 && @printf("%s longest wait with work pending: %.2f s\n", k, worst[k]); end
println("logs: ", isempty(log) ? "no ring timeout, reset, stuck flip or failed commit" : log)
println("kwin_wayland: ", k0 == kwin() ? "same process $k0" : "CHANGED $k0 -> $(kwin())")
pass = !cancelled && isempty(log) && k0 == kwin() && wall > 10.5 && gfx > 0 && maximum(values(worst)) < STALL
println(pass ? "PASS" : "FAIL")
