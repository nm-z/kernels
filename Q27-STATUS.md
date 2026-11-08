# qwen3.8-27b.jl on sentry's four V340 dies: where it stands

Measured with the dies cool (short runs; a long run heat-soaks dies 0 and 2, see Hardware):

| | prefill tok/s | decode tok/s | idle % |
|---|---|---|---|
| 1 request, MTP | ~90 | ~45-50 | ~15-18 |
| 2 requests, MTP | ~98 | ~67 | ~11 |

Done (git log): persistent cooperative kernel a die a request; resident weights; TurboQuant 3.5-bit KV; MTP drafting
(`spec 0|1`); 2 requests at once in resident slots of 262,144; counted 4-byte prefill exchange; paired 24-bit
decode exchange words; DeltaNet conv fused into the recurrence; per-request DPM lock (`dpm 0|1`); flashd's engine
protocol with in-kernel stop tokens.

Serving: a mux service (codex/PROTOCOL.md) on sentry, `qwen3.8-27b`: mux (`mux.service`, user) starts
`/data/models/q27/qwen3.8-27b.jl` (with `codex/julia/Mux.jl` beside it) in `mux-qwen3.8-27b.scope`; clients at
`http://sentry.local:18100/v1`; development console `/data/models/q27/console.sock`; log
`~/.local/state/mux/qwen3.8-27b.log`. Codex model `qwen3.8-27b`, provider `ai-mux-sentry`.

Open (the goal): idle under 1% (the exchange waits, 20-50 us each, 128 a forward); DFlash2 drafting (the draft
and the vision mmproj are on sentry in `/data/models/qwen3.8-27b-q5/`); vision; a 262,144-position demonstration
under the one-minute test limit; sampling (greedy only); output per run capped at 8,192 tokens (the per-phase
accounting buffers grow with it).

DFlash2 draft (copies: archy `~/models/qwen3.8-27b/dflash-Qwen3.8-27B-Q4_0.gguf`, sentry
`~/models/qwen3.8-27b-q5/`, engi scratch): arch `dflash`, 5 layers, hidden 5120, 32 q / 8 kv heads x 128, FFN 17408,
block 8, non-causal, sliding window 2048 on all layers, rope 64 dims base 1e7, target taps after layers
6,20,34,48,62, conv kernel 2 group 16, selector rank 256 top-16, mask token 248070. 81 tensors, all Q4_0 matrices
(a new GEMV/GEMM format for the engine): fc 25600x5120 (the 5 taps), selector_hidden 5120x256,
selector_predecessor / selector_successor 256x248320, per layer attn_q/k/v/output, attn_conv_proj 5120x1280 +
attn_conv_base 5120x2x2, ffn_conv_proj + ffn_conv_base, ffn gate/up/down, norms; enc.output_norm, output_norm.
Host reference of the draft: `~/src/flashnext/q27.jl` (spec_ref, draft_block_ref; from llama.cpp src/models/dflash.cpp
and common/speculative.cpp). Verify of up to 9 positions a request needs DeltaNet state rollback to any accepted
prefix (MTP keeps one backup; 8 would need per-position states or a recompute of the accepted positions).

Hardware: sentry's `/data` disk (Hitachi 2 TB on ata3) dropped off at 1:50 PM 2026-11-01 (IDENTIFY failed, ext4
remounted read-only); the model files and the engine socket live there. Die 2 (PCI 0f:00) idles at 84-87 C and 48 W with nothing running; die 0 (07:00) runs hot too. The edge
limit on these cards is 85 C, so under load both throttle to 300 MHz and the engine falls to ~11 tok/s, ~50% idle.
