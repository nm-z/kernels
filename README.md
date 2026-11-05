# flashnext

Qwen3.8-Flash-Next inference on 8 Tesla M60 dies, in one Julia file (`flashnext.jl`).

Models:

- Main model: [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
  (`Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2}-of-00002.gguf`)
- Draft model (DFlash2): [incoai/Qwen3.8-Flash-Next-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2-GGUF)
  (`Qwen3.8-Flash-Next-DFlash2-Q4_K_M.gguf`)

State, measurements and open work: [PLAN.md](PLAN.md).

`lfm3-1.7b.jl`: [LiquidAI/LFM3-1.7B-Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base) in BF16 on an RX 7700 XT, KernelAbstractions over AMDGPU.jl (`julia --project=env/amd -t 12 lfm3-1.7b.jl --model DIR --serve SOCK`). `test/lfm3-ref.jl` makes its reference data, `test/lfm3-desktop.jl` watches the desktop on the same card during a request.
