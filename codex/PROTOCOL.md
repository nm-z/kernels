# mux protocol

mux is the one process on a host that serves AI. Clients talk to it over HTTP. Everything AI on the
host — a model engine, a shim over another provider, an embedding server, a training or embedding job —
is a **service**: one process mux starts and stops itself, talking to mux over a Unix socket.

```
client (Codex, TUI, curl) ──HTTP──▶ mux ──unix socket──▶ service process (one per service)
                                      │
                                      ├── starts / stops every service (compose-style, mux.yaml)
                                      └── records every event (events.db)
```

**Shims.** "ai-mux" (the nm-z/ai-mux repo) is the name of the shims: the provider adapters (chatgpt, gemini, grok, …). A shim is a
`kind: shim` service that registers with mux over the Unix socket and speaks the shim side of this protocol (below), so it has no
HTTP server of its own; mux serves the clients. Everything else here, mux included, is "mux".

## Invariants

1. **A service is up exactly when its process is up and registered.** mux starts the process; the
   service becomes `up` when that process registers. If the process exits, the service is down. If a
   service is set down, its process and everything it started is killed: a model's weights are evicted,
   a shim's browser sessions (kwin-mcp and the rest) end with it.
2. **mux down means every service down.** Service processes run in systemd scopes bound to
   `mux.service`; when mux stops or dies, systemd stops them. Nothing serves AI outside mux.
3. **Only mux's own processes register.** A registration whose peer process is not in the scope mux
   started for that service is refused.
4. **The server sets the context window, sampling and budgets.** A client chooses only the model and the
   reasoning effort. Requests that set anything else that steers generation are refused (400; the one exception,
   `max_tokens` on the llama.cpp face, is below). The
   context window is broadcast (`/v1/models`, every stats event); filling it is handled by compaction.
5. **A service streams or it is dropped.** What each kind must stream is below. A service that ends a
   run or job without it is disconnected and marked `failed`.

## Services and the compose file

Each service is a folder `codex/services/NAME/` with its code and `service.yaml`; one process per service.

```yaml
# codex/services/qwen3.8-27b/service.yaml
kind: engine                       # engine | shim | embed | job
display: Qwen3.8-27B               # optional: the name web chat shows (see the llama.cpp face)
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
# ~/.config/mux/mux.yaml — one per host
listen: 0.0.0.0:18100
socket: /run/user/1000/mux.sock
services_dir: /data/mux/services          # a checkout of codex/services
compat_listen: 0.0.0.0:18101                 # optional: the llama.cpp face (below) on its own address
compat_service: qwen3.8-27b                  # the service that face fronts
services:
  qwen3.8-27b: {up: true}                    # desired state; `mux up|down NAME` rewrites it
  bge-small:   {up: true}
```

The process gets `MUX_SOCK` and `MUX_SERVICE=<name>`. States: `down` · `starting` (running, not
registered yet) · `up` · `busy` · `failed` (exited or broke rule 5; `restart: on-failure` retries after
10 s, 20 s, 40 s … 5 min) · `done` (a job that finished).

## Service ↔ mux (Unix socket, one JSON object per line)

Every service first sends `register` with `service`, `kind` and the kind's fields; mux answers
`registered` (with `ctx` for engines and shims) or `error` and closes. `cancel {id}` from mux means stop
that run as soon as possible and then end it normally.

### engine: token ids in, token ids out

mux renders the chat template, tokenizes, detokenizes, splits reasoning from the answer and parses tool
calls; the engine only turns token ids into token ids.

| engine → mux | fields | when |
|---|---|---|
| `register` | `positions` (most it holds), `max_new` (most tokens a run makes), `slots`, `vision` | first |
| `prefill` | `id`, `done`, `total` | at least every 500 ms during the prompt, and at `done == total` |
| `tokens` | `id`, `ids` | as they are made |
| `stats` | `id`, `prefill_tps`, `decode_tps`, `idle_pct` (live) | at least every 500 ms while running |
| `done` | `id`, `prompt`, `generated`, `prefill_tps`, `decode_tps`, `idle_pct`, `stopped` | last |
| `error` | `id`, `message` | a run failed |
| `image` | `img`, `tokens` | reply to `image` |

| mux → engine | fields |
|---|---|
| `run` | `id`, `ids`, `max_new` (`min(ctx - prompt, engine max_new)`), `stop`, `images` (`[{img, at}]`) |
| `image` | `img` (hash), `bytes` (base64): encode and keep it for the next run that uses it |

Refused: unknown service, wrong peer, `positions < ctx`.

### shim: a provider over the network (another API, a web page driven through kwin-mcp)

| shim → mux | fields | when |
|---|---|---|
| `register` | `slots`, `vision`, `auto` (true for a route that picks among providers), optional provider `remaining`, `reset_time`, `tokens_served_today` | first |
| `route` | `id`, `trying` (the route(s) being attempted now, e.g. `["vertex/gemini-3-pro"]`), `failed` (ones given up on) | an auto shim: every time its choice changes |
| `ttft` | `id`, `ms` | when the first token of output arrives |
| `delta` | `id`, `channel` (`reasoning` or `content`), `text` | as it streams |
| `tool_call` | `id`, `call_id`, `name`, `arguments` (JSON text) | each call |
| `capacity` | `remaining`, `reset_time`, `tokens_served_today` (provider-reported; each unavailable value is an empty string) | after registration and whenever provider facts change |
| `done` | `id`, `input_tokens`, `output_tokens`, `total_tokens` (each provider-reported when present; absent values are empty strings in client usage) | last |
| `error` | `id`, `message` | |

Each `capacity` message replaces the shim's complete snapshot; include all three fields and use an
empty string when a provider does not expose one. Mux preserves these values without estimating or
aggregating them. It clears a shim's cached capacity when that shim disconnects or stops. A shim
sends no periodic keep-alive for capacity.

mux → shim: `run` with `id`, `messages` (chat form, system first), `tools`, `effort`, `images`.

### embed: an embedding server

| embed → mux | fields |
|---|---|
| `register` | `dims`, `max_batch`, `device` |
| `vectors` | `id`, `vectors` (one per input, in order) |
| `error` | `id`, `message` |

mux → embed: `run` with `id`, `texts`.

### job: finite work (training, embedding a large set, a backfill)

| job → mux | fields | when |
|---|---|---|
| `register` | `title` | first |
| `progress` | `pct` (0–100), `detail` (what it is on) | at least every 10 s and on every change of `pct` |
| `finished` | `ok`, `message` | last; the process then exits (the service becomes `done`) |

## Clients ↔ mux (HTTP)

| request | answer |
|---|---|
| `POST /v1/responses` | the OpenAI Responses API, streamed (SSE); engine and shim services |
| `POST /v1/chat/completions`, `GET /props` | the llama.cpp-compatible face for web chat, below |
| `POST /v1/embeddings` | the OpenAI embeddings API; embed services |
| `GET /v1/models` | `{"data":[{"id","kind","ctx","efforts","vision","state"}]}` |
| `GET /v1/services` | every service: `name, kind, state, device, pid, ctx, slots, since, last, progress, log, reason` |
| `GET /v1/capacity` | each shim's `source, state, remaining, reset_time, tokens_served_today`; absent provider facts are empty strings |
| `POST /v1/services/NAME/up`, `.../down` | change the desired state (rewrites mux.yaml) |
| `GET /v1/events?since=TS` | SSE: the recorded events since `TS`, then live ones (state changes, runs, stats, routes, job progress) |

Every event is also appended to `~/.local/state/mux/events.db` (SQLite: `events(ts, service, run, type,
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

`response.ai_mux.stats` is the one wire name that keeps its old spelling (`ai_mux`), for client compatibility (Codex reads it); mux's
other names, paths and variables are all `mux`.

## Compaction

The client keeps the conversation; the server owns the window. When a request's prompt would not fit in
`ctx`, mux answers 400 with `"code": "context_length_exceeded"`; a client compacts before that, from
the `ctx` it is given and the `usage` of each response.

To compact, the client sends a normal `POST /v1/responses` whose `input` is the conversation followed by
one `{"type":"compaction_trigger"}` item (Codex's remote compaction). mux has the model summarize the
conversation (dropping the oldest turns first if even that prompt does not fit) and answers with one
`compaction` item, `{"type":"compaction","encrypted_content": TEXT}`, where TEXT is opaque to the client
(mux's own encoding of the summary). The client replaces the compacted history with that item, keeping
whatever it retains after it (Codex keeps its recent user messages). In later requests mux expands a
`compaction` item into the summary at its place in the conversation.

## llama.cpp-compatible face for web chat

A web chat gateway (nm-z.com/chat's) polls each backend as it would a llama.cpp server and chats with it. mux
answers in that shape, from the same services, the same prompts and the same rules; nothing here changes
what `/v1/responses` does.

| request | answer |
|---|---|
| `GET /props` | `{"default_generation_settings":{"n_ctx":CTX},"modalities":{"vision":BOOL},"chat_template":"","model_path":GGUF}` for the compat service if it is up, else for the first engine that is up (the service name if it has no gguf); 404 if none |
| `GET /v1/models` | only on the compat listener: `{"object":"list","data":[{"id":"<display>.gguf","object":"model"}]}`, the one model named as llama-server names its GGUF (`display` of service.yaml, else the service name); listed whether or not it is up, which `/props` says |
| `POST /v1/chat/completions` | OpenAI Chat Completions, streamed or not, on an engine or shim; `model` is a service name or its display name (with or without `.gguf`), and without one the compat service |

`/props` and chat completions are on the main address too. `GET /v1/models` there stays mux's own (service names are the
ids clients send), so a client that can only be given host:port, as the gateway can, is given `compat_listen`, a second
address that serves only these three paths, in front of `compat_service`.

Chat completions follow rule 4 like `/v1/responses`: `temperature`, `top_p`, `top_k`, `seed`, `presence_penalty`,
`frequency_penalty`, `stop`, `n`, `logit_bias`, `response_format` and an effort not in the service's `efforts` are refused with 400.
**The exception:** `max_tokens` and `max_completion_tokens` are ignored, not refused, because the gateway hard-codes `max_tokens`
and the server owns the output budget (a run ends at `min(ctx - prompt, the engine's max_new)` with `finish_reason: "length"`); the llama.cpp
transport fields `cache_prompt` and `stream_options` (usage is always sent) are ignored too. The effort is `reasoning_effort` or
`effort`, by default the middle of the service's `efforts`. Messages (system, developer, user, assistant, tool; content a string or
parts, images as `image_url` `data:` URLs) and tools go in the OpenAI chat shape and are folded into the same chat template as
`/v1/responses` input; an assistant message's `reasoning_content` is kept. A prompt that does not fit is a 400
`context_length_exceeded`.

The stream is plain chat completions, `data: {…}` lines and `data: [DONE]`, with no event names: a first chunk with
the role, `delta.reasoning_content`, `delta.content` and `delta.tool_calls` (each call whole, with its `index`) as they are made, then one
chunk with an empty delta, `finish_reason` (`stop`, `tool_calls`, `length`), `usage` and the final `timings`. A run that fails after the
stream began ends with `data: {"error":{…}}` and `[DONE]`.

`timings` is llama.cpp's object, which web chat reads for its speeds. For an engine, every `response.ai_mux.stats` update (so at most every
250 ms and at least every 500 ms) is also a chunk with an empty delta, `finish_reason: null` and
`"timings":{"prompt_n","prompt_per_second","predicted_n","predicted_per_second","cache_n":0}` (plus `idle_pct` when the engine has one), from the
same numbers: while the prompt runs `prompt_n` is the positions done so far, `prompt_per_second` the live prefill tok/s and the `predicted_*`
are 0; during generation `prompt_n` is the prompt, `prompt_per_second` the prefill tok/s, `predicted_n` the tokens made so far and
`predicted_per_second` the live decode tok/s. The last chunk carries the engine's final measured values in `timings` (`prompt_n` the prompt
tokens, `cache_n` those the engine already had, `predicted_n` the tokens made, both speeds, `idle_pct`) with `prompt_ms` and `predicted_ms`
derived from them, next to `usage`; a `stream:false` answer has the same `timings` next to `usage`. A shim has no such numbers, so its
stream has no `timings`; after 10 s without a chunk it gets an empty-delta chunk instead of an SSE comment.
