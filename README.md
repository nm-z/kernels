**Requirements:**
- one Julia file per model arch
- prompt caching
- prompt prefixes
- KV caching
- max concurrent requests
- model's native: </br>
	- attn mechanism[s] </br>
	- sampling </br>
	- max ctx length </br>
	- chat template </br>
	- image encoder (if applicable) </br>
- one exe run per request
- every kernel dynamically dispatched

`lfm3-1.7b.jl`: [LiquidAI/LFM3-1.7B-Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base) in BF16 on an RX 7700 XT, KernelAbstractions over AMDGPU.jl (`julia --project=env/amd -t 12 lfm3-1.7b.jl --model DIR --serve SOCK`). `test/lfm3-ref.jl` makes its reference data, `test/lfm3-desktop.jl` watches the desktop on the same card during a request.
