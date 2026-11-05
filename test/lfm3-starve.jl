# lfm3-starve.jl: how long the desktop's gfx ring waits under each training kernel: the kernel runs back to back for 20 s at
# each workgroup count while the ring's fences (debugfs) are sampled; a wait is the time the signaled fence stands still with
# work pending (the driver resets the ring at 10 s). Optionally with VRAM filled first, to see the eviction case.
#   julia --project=env/amd -t 12 test/lfm3-starve.jl DIR [gemv,gtt,gemm,conv,soft] [27,54,108] [extra VRAM GB]
include(joinpath(@__DIR__, "..", "lfm3-1.7b.jl"))
dir = ARGS[1]; which = split(length(ARGS) > 1 ? ARGS[2] : "gemv,gtt,gemm,conv,soft", ","); nwgs = parse.(Int, split(length(ARGS) > 2 ? ARGS[3] : "27,54,108", ",")); extra = length(ARGS) > 3 ? parse(Float64, ARGS[4]) : 0.0
AMDGPU.stream!(AMDGPU.HIPStream(:low))
M, g = load_engine(dir; caches = false)
KTR[] = k_tr!(ROCBackend(), Int(WGS)); N = 1536
xp = AMDGPU.zeros(Float32, N * NFF ÷ 2); out = AMDGPU.zeros(Float32, N * NFF); x = AMDGPU.rand(Float32, N * 3NE); y = AMDGPU.zeros(Float32, N * NE); P = AMDGPU.rand(Float32, NH * N * N)
fill = extra > 0 ? AMDGPU.rand(Float32, round(Int, extra * 2^28)) : nothing
println("VRAM used: $(round(parse(Int, read("/sys/class/drm/card1/device/mem_info_vram_used", String)) / 2^30; digits = 2)) GB")
fence_start()
function run(name, f)
    worst = 0.0; samples = 0; s0, ts = UInt64(0), time(); t0 = time(); n = 0
    while time() - t0 < 20
        f(); n += 1; s, e = fence(); now = time(); samples += 1
        if s != s0 || s == e
            s0, ts = s, now
        else
            worst = max(worst, now - ts)
        end
    end
    AMDGPU.synchronize()
    @printf("%-28s %6d in 20 s, gfx ring's longest wait %.2f s (%d samples)\n", name, n, worst, samples); flush(stdout)
end
w1 = M.L[1].M[6]; taps = M.L[1].V[5]
for nwg in nwgs
    "gemv" in which && run("gemv w1 nwg $nwg", () -> (trun!(T_PACK, N; nwg = nwg, i1 = NE, i2 = NE, a = dp(x), c = dp(xp)); trun!(T_GEMV, N; nwg = nwg, i1 = NFF ÷ 32, i3 = NE, i4 = NFF, c = dp(xp), d = dp(out), W = wp(w1))))
    "conv" in which && run("conv2 nwg $nwg", () -> trun!(T_CONV2, N; nwg = nwg, a = dp(x), b = dp(out), d = dp(taps), c = dp(y), e = dp(out)))
    "soft" in which && run("softmax nwg $nwg", () -> trun!(T_SOFT, N; nwg = nwg, i1 = 0, a = dp(P)))
    if "gtt" in which
        h, d = pinned(4 * NFF * NE); a = AMDGPU.rand(Float32, NFF * NE)
        run("gtt axpby nwg $nwg", () -> axpby!(dp(a), dp(d), dp(d), NFF, NE, 1, 0.9))
    end
end
if "gemm" in which
    A = AMDGPU.rand(Float32, NFF, NE); Bm = AMDGPU.rand(Float32, N, NE); C = AMDGPU.zeros(Float32, NFF, N)
    run("rocblas gemm", () -> sgemm!(false, true, NFF, N, NE, 1, pointer(A), NFF, pointer(Bm), N, 0, pointer(C), NFF))
end
