# lfm3-mapped.jl: the kernel's progress words reach the host while it runs (pinned mapped memory, tick)
include(joinpath(@__DIR__, "..", "lfm3.jl"))
dev!(1); p, d = pinned(4096); dp = lp(UInt32, d)
@kernel unsafe_indices = true function kw!(dp::LP{UInt32})
    tid = Int32(KI.get_local_id(Int32).x - 1); gid = Int32(KI.get_group_id(Int32).x - 1)
    if gid == 0 && tid == 0
        t0 = tick(); atomst!(dp, UInt32(1)); x = UInt32(0)
        for i in 1:20; i2 = Int64(0); while i2 < 50_000_000; x += UInt32(i2 & 7); i2 += 1; end; atomst!(dp + 4, UInt32(i)); end
        atomst!(dp + 8, UInt32((tick() - t0) >> 20)); atomst!(dp + 12, x)
    end
end
dev!(1); k = kw!(BE, 256); k(dp; ndrange = 256 * 64); hp = Ptr{UInt32}(p)
for i in 1:40; sleep(0.25); @printf("%.2f s: %d %d\n", 0.25i, unsafe_load(hp, 1), unsafe_load(hp, 2)); unsafe_load(hp, 3) != 0 && break; end
sync(); @printf("done: %d %d ticks>>20 %d x %d\n", unsafe_load(hp, 1), unsafe_load(hp, 2), unsafe_load(hp, 3), unsafe_load(hp, 4))
