# lfm3-desktop.jl: do the screens on the same card keep updating while lfm3-1.7b.jl runs one long dispatch? Before and during
# a chat request it reads from debugfs (needs sudo) each ring's fences and each CRTC's scanned-out framebuffers, and from
# the logs any stuck page flip or KWin commit that fails. The rings alone say nothing about the screens: after a GPU reset
# the rings ran while two outputs' flips stayed stuck. Any of these cancels the request at once: a ring with work pending
# whose fence stands still for STALL seconds (the ring timeouts are 10 s), a failed commit, a flip timeout, a reset.
# PASS needs: a dispatch longer than the ring timeouts, no cancel, every CRTC that changed framebuffers in the 5 s before the
# request changing them during it, clean logs, the same kwin_wayland process. What a screen shows is for its owner to check.
#   julia test/lfm3-desktop.jl SOCK REF.ids NTOKENS
using Sockets, Printf
const STALL = 3.0
const BAD = "ring .*timeout|GPU reset|GPU Hang|flip_done|commit wait"
sock, ids, n = ARGS[1], ARGS[2], parse(Int, ARGS[3])
kwin() = strip(read(`pgrep -o -x kwin_wayland`, String))
const DRI = first(d for d in readlines(`sudo sh -c 'ls -d /sys/kernel/debug/dri/*/'`) if occursin("gfx_0.0.0", read(`sudo sh -c "cat $(d)amdgpu_fence_info 2>/dev/null; true"`, String)))
function fenceinfo()
    r = Dict{String,Tuple{UInt64,UInt64}}()
    for m in eachmatch(r"--- ring \d+ \(([^)]+)\) ---\nLast signaled fence\s+0x([0-9a-f]+)\nLast emitted\s+0x([0-9a-f]+)", read(`sudo cat $(DRI)amdgpu_fence_info`, String))
        r[m[1]] = (parse(UInt64, m[2], base = 16), parse(UInt64, m[3], base = 16))
    end
    r
end
function fbs!(seen::Dict{String,Set{Int}})                 # each CRTC's framebuffers now, into seen
    for m in eachmatch(r"crtc=(crtc-\d+)\n\s*fb=([1-9]\d*)", read(`sudo cat $(DRI)state`, String))
        push!(get!(seen, m[1], Set{Int}()), parse(Int, m[2]))
    end
end
badlog(since) = read(pipeline(ignorestatus(`journalctl -k --since $since`), ignorestatus(`rg -i $BAD`)), String)
ebusy(since) = (s = strip(read(pipeline(ignorestatus(`journalctl --user --since $since`), ignorestatus(`rg -c "Atomic commit failed"`)), String)); isempty(s) ? 0 : parse(Int, s))
k0 = kwin(); since = strip(read(`date '+%Y-%m-%d %H:%M:%S'`, String)); sleep(1)
isempty(badlog(since)) && ebusy(since) == 0 || (println("FAIL: the logs are not clean before the request"); exit(1))
before = Dict{String,Set{Int}}(); b0 = fenceinfo()["gfx_0.0.0"][1]; tb = time()
while time() - tb < 5; fbs!(before); sleep(0.1); end
base = (fenceinfo()["gfx_0.0.0"][1] - b0) / (time() - tb)
r0 = fenceinfo(); rings = [k for k in keys(r0) if startswith(k, "gfx") || startswith(k, "comp")]
c = connect(sock); ii = Int.(reinterpret(Int32, read(ids)))[1:64]
println(c, "cancel_on_eof 1\nchat ", n, " ", join(ii, ' ')); flush(c)
toks = Ref(0); done = Ref(false); line = Ref("")
reader = @async try
    while !eof(c)
        l = readline(c); startswith(l, "tok") && (toks[] += 1)
        startswith(l, "done") && (line[] = l; done[] = true; break)
    end
catch
    done[] = true
end
last = Dict(k => (r0[k][1], time()) for k in rings); worst = Dict(k => 0.0 for k in rings); t0 = time(); why = ""; samples = 0; tl = t0
during = Dict{String,Set{Int}}()
while !done[] && time() - t0 < 58
    r = fenceinfo(); now = time(); global samples += 1; fbs!(during)
    for k in rings
        s, e = r[k]; a, ta = last[k]
        if s != a || s == e
            last[k] = (s, now)
        else
            worst[k] = max(worst[k], now - ta)
            now - ta > STALL && isempty(why) && (global why = "$k stood still with work pending")
        end
    end
    if now - tl > 0.5                                      # (the logs, a couple of times a second)
        global tl = now
        isempty(why) && !isempty(badlog(since)) && (global why = "the kernel log")
        isempty(why) && ebusy(since) > 0 && (global why = "kwin's atomic commits failed")
    end
    if !isempty(why)
        println("CANCELLED: ", why); close(c); break
    end
    sleep(0.1)
end
wall = time() - t0; isopen(c) && close(c)
r1 = fenceinfo(); sleep(12)                                # (a stuck flip reports after 10 s)
lg = badlog(since); eb = ebusy(since)
gfx = r1["gfx_0.0.0"][1] - r0["gfx_0.0.0"][1]
@printf("wall %.1f s, %d tokens, %d polls, %s\n", wall, toks[], samples, line[])
@printf("gfx_0.0.0 fences: %.1f a second during the run, %.1f before it\n", gfx / wall, base)
for k in sort(rings); worst[k] > 0 && @printf("%s longest wait with work pending: %.2f s\n", k, worst[k]); end
stuck = String[]
for k in sort(collect(keys(before)))
    a = length(before[k]); b = length(get(during, k, Set{Int}()))
    @printf("%s framebuffers seen: %d before, %d during\n", k, a, b)
    a > 1 && b <= 1 && push!(stuck, k)
end
println("logs after 12 s more: ", isempty(lg) && eb == 0 ? "no ring timeout, reset, stuck flip or failed commit" : "$eb failed commits\n$lg")
println("kwin_wayland: ", k0 == kwin() ? "same process $k0" : "CHANGED $k0 -> $(kwin())")
pass = isempty(why) && isempty(lg) && eb == 0 && isempty(stuck) && k0 == kwin() && wall > 10.5
println(pass ? "PASS" : "FAIL")
