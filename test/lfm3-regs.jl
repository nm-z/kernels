# lfm3-regs.jl: the kernel's registers, local memory and blocks per SM, with no model loaded
#   julia --project=env/cuda test/lfm3-regs.jl
include(joinpath(@__DIR__, "..", "lfm3.jl"))
A = RQ([T <: Core.LLVMPtr ? Core.Intrinsics.bitcast(T, UInt64(0)) : zero(T) for T in fieldtypes(RQ)]...)
k = krun!(BE, 256); nd_, wg, it, _ = KernelAbstractions.launch_config(k, 256, nothing)
launch = KernelAbstractions.select_launch(KernelAbstractions.sizes_only(k), wg, it)
ctx = KernelAbstractions.mkcontext(k, nd_, it, launch; subgroups = KernelAbstractions.kernel_subgroups(k))
_, tt = KernelAbstractions.prepare_arguments(BE, k.f, ctx, (A,))
kf = KI.kernel_function(BE, k.f, tt; KernelAbstractions.compiler_options(k)...); at = CUDA.attributes(kf.kern.kernel.fun)
@printf("kernel: %d registers, %d B local, %d B static smem, %d blocks/SM\n", at[CUDA.FUNC_ATTRIBUTE_NUM_REGS], at[CUDA.FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES], at[CUDA.FUNC_ATTRIBUTE_SHARED_SIZE_BYTES], CUDA.active_blocks(kf.kern.kernel.fun, 256))
