# Using ai-mux

Point any Responses API client at `http://HOST:18100/v1`. A client chooses the model (an endpoint's name) and
the reasoning effort; the server sets context, sampling and output budget, and refuses (400) `temperature`,
`top_p`, `top_k`, `seed`, `max_output_tokens`, `presence_penalty`, `frequency_penalty`, `stop` and an effort
the endpoint does not offer.

## Codex (`~/.codex/config.toml`)

    model = "qwen3.8-27b"
    model_provider = "ai-mux"
    model_reasoning_effort = "high"         # none low medium high xhigh, as the endpoint's `efforts`

    [model_providers.ai-mux]
    name = "ai-mux"
    base_url = "http://HOST:18100/v1"
    wire_api = "responses"
    stream_idle_timeout_ms = 600000          # a long prompt prefills for minutes before the first token

Function and custom (freeform, e.g. `apply_patch`) tools both work; custom calls come back as
`custom_tool_call` items. Images are `input_image` parts with `data:` URLs (vision endpoints only).
A run that reaches the server's output budget ends with `response.incomplete` (`max_output_tokens`).

## Extra stream event

While the engine runs, and once more with the final numbers before `response.completed`:

    event: response.ai_mux.stats
    data: {"type":"response.ai_mux.stats","phase":"prefill"|"decode","prefill":{"done":N,"total":N,"tps":X},"decode":{"tokens":N,"tps":X},"ctx":N}

at most every 250 ms and at least every 500 ms; the final one adds `"idle_pct"`. Other clients ignore it.
`: keepalive` comments go out every 10 s of silence (queued requests wait for a free slot).

## Other endpoints

    GET  /v1/models                  {"data":[{"id","ctx","efforts","vision","state"}]}
    GET  /v1/endpoints               {"data":[{name,state,device,pid,ctx,slots,since,last,log,reason,…}]}
    POST /v1/endpoints/NAME/up|down  change the desired state (rewrites the compose file)
    GET  /v1/events                  SSE for any number of observers: endpoint.state, run.start, run.prefill,
                                     run.stats, run.tokens (counts, never token text), run.end

    curl -N http://HOST:18100/v1/responses -d '{"model":"qwen3.8-27b","stream":true,"input":"hi"}'
