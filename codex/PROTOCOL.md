# ai-mux protocol

ai-mux is the one process on a host that serves AI. Clients talk to it over HTTP; every model it serves
is an **endpoint** it starts and stops itself, and that endpoint's **engine** talks to it over a Unix
socket. Nothing else on the host serves AI.

```
client (Codex, TUI, curl) ──HTTP──▶ ai-mux ──unix socket──▶ engine (a kernel's process)
                                      │
                                      └── starts / stops every endpoint (ai-mux.yaml)
```

## Invariants

1. **An endpoint is up exactly when its process is up and registered.** ai-mux starts the process; the
   endpoint becomes `up` when that process registers. If the process exits, the endpoint is `down`. If
   an endpoint is set down, its process (and every process it started) is killed, which evicts its
   weights.
2. **ai-mux down means every endpoint down.** Endpoint processes run in systemd scopes bound to
   `ai-mux.service`; when ai-mux stops or dies, systemd stops them.
3. **Only ai-mux's own processes register.** A registration whose peer process is not in the scope
   ai-mux started for that endpoint is refused.
4. **The server sets the context window, sampling and output budget.** A client chooses only the model
   and the reasoning effort. Requests that set anything else that steers generation are refused (400).
5. **An engine streams or it is dropped.** Every run must stream its tokens as they are made, prefill
   progress during the prompt, and decode stats during generation, and end with its measured prefill
   tok/s, decode tok/s and idle %. An engine that finishes a run without these is disconnected and its
   endpoint marked `failed`.

## ai-mux.yaml (one per host)

```yaml
listen: 0.0.0.0:18100              # HTTP for clients
socket: /run/user/1000/ai-mux.sock # engines connect here (AI_MUX_SOCK)
endpoints:
  qwen3.8-27b:
    up: true                       # desired state; `ai-mux up|down NAME` changes it
    gguf: /data/models/qwen3.8-27b-q5/Qwen3.8-27B.Q5_K_M.gguf   # tokenizer + chat template
    ctx: 262144                    # context window, positions (prompt + output)
    efforts: [none, low, medium, high, xhigh]
    vision: false
    device: sentry 4x Radeon Pro V340 (gfx900)
    command: [julia, -t, "24", qwen3.8-27b.jl, --model, /data/models/qwen3.8-27b-q5/Qwen3.8-27B.Q5_K_M.gguf]
    cwd: /data/models
    env: {JULIA_DEPOT_PATH: /data/julia-depot}
    post_stop: [sh, -c, "for f in /sys/class/drm/card*/device/power_dpm_force_performance_level; do echo auto > $f; done"]
    restart: on-failure            # or: no
```

The endpoint process gets `AI_MUX_SOCK` and `AI_MUX_ENDPOINT=<name>` in its environment.

States: `down` (desired down, nothing running) · `starting` (process running, not registered: loading
weights, compiling) · `up` (registered, idle) · `busy` (running a request) · `failed` (exited, or broke
rule 5; restarts if `restart: on-failure`, with 10 s, 20 s, 40 s … up to 5 min between tries).

## Engine ↔ ai-mux (Unix socket, one JSON object per line)

The engine connects to `AI_MUX_SOCK` and sends `register`. Token ids are the GGUF vocabulary's.
ai-mux renders the chat template, tokenizes, detokenizes, splits reasoning from the answer and parses
tool calls; the engine only turns token ids into token ids.

Engine → ai-mux

| op | fields | when |
|---|---|---|
| `register` | `endpoint`, `positions` (most the engine holds), `max_new` (most tokens one run generates), `slots` (requests at once), `vision` | first line |
| `prefill` | `id`, `done`, `total` (prompt positions run so far / to run) | at least every 500 ms while the prompt runs, and once at `done == total` |
| `tokens` | `id`, `ids` (new token ids, in order) | as the engine makes them |
| `stats` | `id`, `prefill_tps`, `decode_tps` | at least every 500 ms while it runs (live values) |
| `done` | `id`, `prompt`, `generated`, `prefill_tps`, `decode_tps`, `idle_pct`, `stopped` (ended on a stop token) | last line of a run |
| `error` | `id`, `message` | a run failed; no `done` follows |
| `image` | `img`, `tokens` (positions it takes) | reply to an `image` |

ai-mux → engine

| op | fields |
|---|---|
| `registered` | `ctx` (the endpoint's configured window; the engine must not run past it) |
| `run` | `id`, `ids` (the prompt), `max_new`, `stop` (ids that end the run), `images` (`[{img, at}]`: the image `img` fills positions `at` …) |
| `cancel` | `id` (stop as soon as possible, then send `done`) |
| `image` | `img` (a hash), `bytes` (base64): encode it and keep it until the next run that uses it |

ai-mux refuses `register` if the endpoint is unknown, the peer is not that endpoint's process, or
`positions < ctx`. A run's `max_new` is `min(ctx - prompt, the engine's max_new)`.

## Clients ↔ ai-mux (HTTP)

| request | answer |
|---|---|
| `POST /v1/responses` | the OpenAI Responses API, streamed (SSE); see below |
| `GET /v1/models` | `{"data":[{"id", "ctx", "efforts", "vision", "state"}]}` |
| `GET /v1/endpoints` | every endpoint: `name, state, device, pid, ctx, slots, since, last` (last run's prefill tok/s, decode tok/s, idle %) |
| `POST /v1/endpoints/NAME/up`, `.../down` | change the desired state (also rewrites `up:` in ai-mux.yaml) |
| `GET /v1/events` | SSE of everything every endpoint does: state changes, runs, prefill/decode stats, tokens counts |

`POST /v1/responses` accepts `model`, `input`, `instructions`, `tools`, `reasoning.effort`, `stream`
and the transport fields Codex sends (`tool_choice`, `parallel_tool_calls`, `store`, `include`,
`prompt_cache_key`, `text`, `client_metadata`, `service_tier`, `stream_options`). `temperature`,
`top_p`, `top_k`, `seed`, `max_output_tokens`, `presence_penalty`, `frequency_penalty`, `stop`, and
an effort not in the endpoint's `efforts` are refused with 400.

Images are `input_image` parts with `data:` URLs. Function and custom (freeform) tools both work;
custom tool calls come back as `custom_tool_call` items.

The stream is the Responses API's (`response.created`, `response.output_item.added/done`,
`response.reasoning_text.delta`, `response.output_text.delta`, `response.completed` with `usage`),
with reasoning streamed as it is generated, plus one more event type:

```
event: response.ai_mux.stats
data: {"type":"response.ai_mux.stats","phase":"prefill"|"decode","prefill":{"done":N,"total":N,"tps":X},"decode":{"tokens":N,"tps":X},"ctx":N}
```

sent at least every 500 ms while the engine runs, and once with the final measured values (plus
`"idle_pct"`) before `response.completed`.
