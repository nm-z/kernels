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

Serving: `/data/models/q27v.jl` (this file), socket `/data/models/q27.sock`; flashd (`/data/flashd/kern`, patched:
folds system/developer messages, parses string tool-call arguments) on 127.0.0.1:18096; units `q27-engine`,
`q27-flashd` (sentry, user); engi: `codex-q27-tunnel`, route in `~/.config/codex-flash-next/routes.toml`; Codex model
`sentry-loc/Qwen3.8-27B`.

Open (the goal): idle under 1% (the exchange waits, 20-50 us each, 128 a forward); DFlash2 drafting
(dflash-Qwen3.8-27B-Q4_0.gguf, not downloaded yet); vision (mmproj-Qwen3.8-27B-Q8_0.gguf, not downloaded); a
262,144-position demonstration under the one-minute test limit; sampling (greedy only).

Hardware: die 2 (PCI 0f:00) idles at 84-87 C and 48 W with nothing running; die 0 (07:00) runs hot too. The edge
limit on these cards is 85 C, so under load both throttle to 300 MHz and the engine falls to ~11 tok/s, ~50% idle.
