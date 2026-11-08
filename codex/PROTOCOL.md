# ai-mux protocol

ai-mux is the one process on a host that serves AI. Clients talk to it over HTTP. Everything AI on the
host — a model engine, a shim over another provider, an embedding server, a training or embedding job —
is a **service**: one process ai-mux starts and stops itself, talking to ai-mux over a Unix socket.

```
client (Codex, TUI, curl) ──HTTP──▶ ai-mux ──unix socket──▶ service process (one per service)
                                      │
                                      ├── starts / stops every service (compose-style, ai-mux.yaml)
                                      └── records every event (events.db)
```

## Invariants

1. **A service is up exactly when its process is up and registered.** ai-mux starts the process; the
   service becomes `up` when that process registers. If the process exits, the service is down. If a
   service is set down, its process and everything it started is killed: a model's weights are evicted,
   a shim's browser sessions (kwin-mcp and the rest) end with it.
2. **ai-mux down means every service down.** Service processes run in systemd scopes bound to
   `ai-mux.service`; when ai-mux stops or dies, systemd stops them. Nothing serves AI outside ai-mux.
3. **Only ai-mux's own processes register.** A registration whose peer process is not in the scope ai-mux
   started for that service is refused.
4. **The server sets the context window, sampling and budgets.** A client chooses only the model and the
   reasoning effort. Requests that set anything else that steers generation are refused (400). The
   context window is broadcast (`/v1/models`, every stats event); filling it is handled by compaction.
5. **A service streams or it is dropped.** What each kind must stream is below. A service that ends a
   run or job without it is disconnected and marked `failed`.

## Services and the compose file

Each service is a folder `codex/services/NAME/` with its code and `service.yaml`; one process per service.

```yaml
# codex/services/qwen3.8-27b/service.yaml
kind: engine                       # engine | shim | embed | job
gguf: /data/models/qwen3.8-27b-q5/Qwen3.8-27B.Q5_K_M.gguf   # engine: tokenizer + chat template
ctx: 262144                        # engine, shim: context window broadcast to clients
efforts: [none, low, medium, high, xhigh]
vision: false
device: sentry 4x Radeon Pro V340 (gfx900)
command: [/usr/bin/julia, -t, "24", qwen3.8-27b.jl, --model, /data/models/qwen3.8-27b-q5/Qwen3.8-27B.Q5_K_M.gguf]
cwd: /data/models/q27
env: {JULIA_DEPOT_PATH: /data/julia-depot}
post_stop: [sh, -c, "for f in /sys/class/drm/card*/device/power_dpm_force_performance_level; do echo auto > $f; done"]
restart: on-failure                # or: no (jobs: no)
```

```yaml
# ~/.config/ai-mux/ai-mux.yaml — one per host
listen: 0.0.0.0:18100
socket: /run/user/1000/ai-mux.sock
services_dir: /data/ai-mux/services          # a checkout of codex/services
services:
  qwen3.8-27b: {up: true}                    # desired state; `ai-mux up|down NAME` rewrites it
  bge-small:   {up: true}
```

The process gets `AI_MUX_SOCK` and `AI_MUX_SERVICE=<name>`. States: `down` · `starting` (running, not
registered yet) · `up` · `busy` · `failed` (exited or broke rule 5; `restart: on-failure` retries after
10 s, 20 s, 40 s … 5 min) · `done` (a job that finished).

## Service ↔ ai-mux (Unix socket, one JSON object per line)

Every service first sends `register` with `service`, `kind` and the kind's fields; ai-mux answers
`registered` (with `ctx` for engines and shims) or `error` and closes. `cancel {id}` from ai-mux means stop
that run as soon as possible and then end it normally.

### engine: token ids in, token ids out

ai-mux renders the chat template, tokenizes, detokenizes, splits reasoning from the answer and parses tool
calls; the engine only turns token ids into token ids.

| engine → ai-mux | fields | when |
|---|---|---|
| `register` | `positions` (most it holds), `max_new` (most tokens a run makes), `slots`, `vision` | first |
| `prefill` | `id`, `done`, `total` | at least every 500 ms during the prompt, and at `done == total` |
| `tokens` | `id`, `ids` | as they are made |
| `stats` | `id`, `prefill_tps`, `decode_tps`, `idle_pct` (live) | at least every 500 ms while running |
| `done` | `id`, `prompt`, `generated`, `prefill_tps`, `decode_tps`, `idle_pct`, `stopped` | last |
| `error` | `id`, `message` | a run failed |
| `image` | `img`, `tokens` | reply to `image` |

| ai-mux → engine | fields |
|---|---|
| `run` | `id`, `ids`, `max_new` (`min(ctx - prompt, engine max_new)`), `stop`, `images` (`[{img, at}]`) |
| `image` | `img` (hash), `bytes` (base64): encode and keep it for the next run that uses it |

Refused: unknown service, wrong peer, `positions < ctx`.

### shim: a provider over the network (another API, a web page driven through kwin-mcp)

| shim → ai-mux | fields | when |
|---|---|---|
| `register` | `slots`, `vision`, `auto` (true for a route that picks among providers) | first |
| `route` | `id`, `trying` (the route(s) being attempted now, e.g. `["vertex/gemini-3-pro"]`), `failed` (ones given up on) | an auto shim: every time its choice changes |
| `ttft` | `id`, `ms` | when the first token of output arrives |
| `delta` | `id`, `channel` (`reasoning` or `content`), `text` | as it streams |
| `tool_call` | `id`, `call_id`, `name`, `arguments` (JSON text) | each call |
| `done` | `id`, `input_tokens`, `output_tokens` (the provider's, when it reports them) | last |
| `error` | `id`, `message` | |

ai-mux → shim: `run` with `id`, `messages` (chat form, system first), `tools`, `effort`, `images`.

### embed: an embedding server

| embed → ai-mux | fields |
|---|---|
| `register` | `dims`, `max_batch`, `device` |
| `vectors` | `id`, `vectors` (one per input, in order) |
| `error` | `id`, `message` |

ai-mux → embed: `run` with `id`, `texts`.

### job: finite work (training, embedding a large set, a backfill)

| job → ai-mux | fields | when |
|---|---|---|
| `register` | `title` | first |
| `progress` | `pct` (0–100), `detail` (what it is on) | at least every 10 s and on every change of `pct` |
| `finished` | `ok`, `message` | last; the process then exits (the service becomes `done`) |

## Clients ↔ ai-mux (HTTP)

| request | answer |
|---|---|
| `POST /v1/responses` | the OpenAI Responses API, streamed (SSE); engine and shim services |
| `POST /v1/embeddings` | the OpenAI embeddings API; embed services |
| `GET /v1/models` | `{"data":[{"id","kind","ctx","efforts","vision","state"}]}` |
| `GET /v1/services` | every service: `name, kind, state, device, pid, ctx, slots, since, last, progress, log, reason` |
| `POST /v1/services/NAME/up`, `.../down` | change the desired state (rewrites ai-mux.yaml) |
| `GET /v1/events?since=TS` | SSE: the recorded events since `TS`, then live ones (state changes, runs, stats, routes, job progress) |

Every event is also appended to `~/.local/state/ai-mux/events.db` (SQLite: `events(ts, service, run, type,
json)`), the one place to look back at what ran.

`POST /v1/responses` accepts `model`, `input`, `instructions`, `tools`, `reasoning.effort`, `stream` and the
transport fields Codex sends (`tool_choice`, `parallel_tool_calls`, `store`, `include`, `prompt_cache_key`,
`text`, `client_metadata`, `service_tier`, `stream_options`). `temperature`, `top_p`, `top_k`, `seed`,
`max_output_tokens`, `presence_penalty`, `frequency_penalty`, `stop` and an effort not in the service's
`efforts` are refused with 400. Images are `input_image` parts with `data:` URLs. Function and custom
(freeform) tools both work. The stream is the Responses API's, with reasoning streamed as
`response.reasoning_text.delta` as it is generated, plus:

```
event: response.ai_mux.stats
data: {"type":"response.ai_mux.stats","phase":"prefill"|"decode","prefill":{"done":N,"total":N,"tps":X},"decode":{"tokens":N,"tps":X},"idle_pct":X,"ctx":N}
```

for engines, at least every 500 ms while running (live values, idle % of the run so far) and once with the final measured values; for
shims, `{"type":"response.ai_mux.stats","phase":"route"|"output","route":{"trying":[…],"failed":[…]},"ttft_ms":N,"ctx":N}`
on every route change and once at the first token.

## Compaction

The client keeps the conversation; the server owns the window. When a request's prompt would not fit in
`ctx`, ai-mux answers 400 with `"code": "context_length_exceeded"`; a client compacts before that, from
the `ctx` it is given and the `usage` of each response.

To compact, the client sends a normal `POST /v1/responses` whose `input` is the conversation followed by
one `{"type":"compaction_trigger"}` item (Codex's remote compaction). ai-mux has the model summarize the
conversation (dropping the oldest turns first if even that prompt does not fit) and answers with one
`compaction` item, `{"type":"compaction","encrypted_content": TEXT}`, where TEXT is opaque to the client
(ai-mux's own encoding of the summary). The client replaces the compacted history with that item, keeping
whatever it retains after it (Codex keeps its recent user messages). In later requests ai-mux expands a
`compaction` item into the summary at its place in the conversation.
