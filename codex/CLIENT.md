# Using ai-mux

Point any Responses API client at `http://HOST:18100/v1`. A client chooses the model (a service's name) and the
reasoning effort; the server sets context, sampling and budgets, and refuses (400) `temperature`, `top_p`,
`top_k`, `seed`, `max_output_tokens`, `presence_penalty`, `frequency_penalty`, `stop` and an effort the
service does not offer.

## Codex (`~/.codex/config.toml`)

    model = "qwen3.8-27b"
    model_provider = "ai-mux"
    model_reasoning_effort = "high"         # none low medium high xhigh, as the service's `efforts`

    [model_providers.ai-mux]
    name = "ai-mux"
    base_url = "http://HOST:18100/v1"
    wire_api = "responses"
    stream_idle_timeout_ms = 600000          # a long prompt prefills for minutes before the first token

Function and custom (freeform, e.g. `apply_patch`) tools work; images are `input_image` parts with `data:` URLs.
A run that reaches the output budget ends with `response.incomplete` (`max_output_tokens`).

## The window and compaction

`GET /v1/models` gives each model's `ctx`; a response's `usage` says how full it is; a prompt that does not
fit is a 400 `context_length_exceeded`. To compact, send the conversation plus `{"type":"compaction_trigger"}`
as the last input item: the answer is one `{"type":"compaction","id":"cmp_…","encrypted_content":…}` item
(a summary, opaque to you). Keep it in the history in place of what it summarizes; ai-mux expands it back into
the summary in later requests.

## Extra stream event: `response.ai_mux.stats`

Engines: `{"phase":"prefill"|"decode","prefill":{"done","total","tps"},"decode":{"tokens","tps"},"idle_pct","ctx"}`,
at most every 250 ms and at least every 500 ms (`idle_pct` is null until the engine has one), and once more with
the final numbers before `response.completed`. Shims: `{"phase":"route"|"output","route":{"trying","failed"},"ttft_ms","ctx"}`
on each route change (auto shims) and at the first token. `: keepalive` comments go out every 10 s of silence.

## Other endpoints

    POST /v1/chat/completions        OpenAI Chat Completions (stream or not; tools; reasoning_effort) on an engine or shim;
                                     max_tokens is ignored: the server owns the output budget
    GET  /props                      {"default_generation_settings":{"n_ctx"},"modalities":{"vision"},…}: llama.cpp's, for web chat
    POST /v1/embeddings              OpenAI embeddings on an embed service (string or strings; float or base64)
    GET  /v1/models                  {"data":[{"id","kind","ctx","efforts","vision","state"}]}   (embed: "dims")
    GET  /v1/services                {"data":[{name,kind,state,device,pid,ctx,slots,since,last,progress,log,reason}]}
    POST /v1/services/NAME/up|down   change the desired state (rewrites the host file)
    GET  /v1/events[?since=TS]       SSE for any number of observers: the recorded events since TS, then live (ids
                                     resume with Last-Event-ID): service.state, run.*, job.progress/finished; never token text
