# slots.jl: how many of the kernel's workgroups fit next to what else the card is running. Every PERIOD it launches NWG
# workgroups with lfm3-1.7b.jl's register footprint (a clobbered v(VG - 1)), each held for HOLD and then gone, and records
# each one's arrival. A workgroup arriving within HOLD of the first got a slot without waiting for one of its own to
# leave; the shortfall from what fits on an idle card is the slots others were holding. The card is held HOLD / PERIOD
# of the time, in short jobs.
#   julia --project=env/amd test/slots.jl SECONDS [NWG=108] [VG=128]
using AMDGPU, KernelAbstractions, Printf
const KI = KernelAbstractions.KernelInterface
const HOLD, PERIOD = 200_000, 0.1                          # 2 ms of the 100 MHz clock; a launch every 100 ms
secs = parse(Int, ARGS[1]); nwg = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 108; vg = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 128
@inline memtime() = Base.llvmcall(("""
    declare i64 @llvm.amdgcn.s.sendmsg.rtn.i64(i32 immarg)
    define i64 @mt() #0 {
        %v = call i64 @llvm.amdgcn.s.sendmsg.rtn.i64(i32 131)
        ret i64 %v
    }
    attributes #0 = { alwaysinline }
    """, "mt"), UInt64, Tuple{})
const CLOBBER = "call void asm sideeffect \"\", \"~{v$(vg - 1)}\"()\nret void"
@eval @inline regs() = Base.llvmcall($CLOBBER, Cvoid, Tuple{})
@kernel unsafe_indices = true function k_slot!(o, h)
    regs()
    if KI.get_local_id(Int32).x == Int32(1)
        t0 = memtime(); o[KI.get_group_id(Int32).x] = t0
        while memtime() - t0 < h; end
    end
    KI.barrier()
end
b = ROCBackend(); o = AMDGPU.zeros(UInt64, nwg); k = k_slot!(b, 256)
k(o, UInt64(10); ndrange = 256); AMDGPU.synchronize()
sdir = "/sys/class/drm/card1/device/gpu_busy_percent"
inh = Int[]; lag = Float64[]; busy = Int[]; t0 = time()
while time() - t0 < secs
    fill!(o, 0x0); k(o, UInt64(HOLD); ndrange = 256nwg); AMDGPU.synchronize()
    a = sort(Array(o)); f = a[1]
    push!(inh, count(t -> t - f < HOLD, a)); push!(lag, (a[end] - f) / 100.0); push!(busy, parse(Int, strip(read(sdir, String))))
    sleep(PERIOD)
end
s = sort(inh); q(p) = s[clamp(round(Int, p * length(s)), 1, length(s))]
@printf("%d launches of %d workgroups (%d VGPRs), %d s; card busy mean %.1f%%, max %d%%\n", length(s), nwg, vg, secs, sum(busy) / length(busy), maximum(busy))
@printf("workgroups in before any of them left: min %d, p1 %d, p10 %d, median %d, p90 %d, max %d\n", s[1], q(0.01), q(0.1), q(0.5), q(0.9), s[end])
@printf("first to last arrival: median %.1f us, max %.1f us\n", sort(lag)[cld(end, 2)], maximum(lag))
@printf("held by others, in workgroups: mean %.1f; when any are held (%d of %d launches), mean %.1f\n", nwg - sum(s) / length(s), count(<(nwg), s), length(s), sum(nwg .- s[s.<nwg]; init = 0) / max(count(<(nwg), s), 1))
