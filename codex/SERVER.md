# Running ai-mux

ai-mux (PROTOCOL.md) is the one process on a host that serves AI. Everything AI here is a **service**: a
process ai-mux starts and stops itself (an engine, a shim, an embed server, a job) that registers over a
Unix socket. Clients use HTTP.

    cd codex && go build -o bin/ai-mux ./cmd/ai-mux && go test ./...

## Services and the host file

`services/NAME/service.yaml` (+ the service's code): `kind` (`engine|shim|embed|job`), `display` (the name web chat shows), `gguf` (engines:
tokenizer + chat template), `ctx`, `efforts`, `vision`, `device`, `command`, `cwd` (default: the folder),
`env`, `post_stop`, `restart` (`on-failure`: 10 s, 20 s … 5 min; jobs default to `no`).
`~/.config/ai-mux/ai-mux.yaml` (example: hosts/sentry.yaml): `listen`, `socket`, `services_dir`,
`services: {NAME: {up: bool}}`, and for web chat `compat_listen` + `compat_service` (a second address that serves only
the llama.cpp face of one service: `/v1/models`, `/props`, `/v1/chat/completions`). The process gets `AI_MUX_SOCK`, `AI_MUX_SERVICE`, `AI_MUX_SERVICE_DIR`.
A service that breaks rule 5 is killed and `failed`: an engine run without prefill lines, stats lines or its
measured `done`; a shim delta before `ttft` (or an auto shim without `route`); wrong embed vectors; a job with
no `progress` for 12.5 s or a `pct` outside 0-100. A shim's `run` has `messages` (chat form; images are
`{"type":"image","img":HASH}` parts), `tools`, `effort`, `images` (`[{img, mime, bytes(base64)}]`).

## Service unit

    install -m755 bin/ai-mux ~/.local/bin/ai-mux
    install -Dm644 ai-mux.service ~/.config/systemd/user/ai-mux.service
    systemctl --user daemon-reload && systemctl --user enable --now ai-mux && loginctl enable-linger $USER

Each service runs in `ai-mux-<name>.scope` (`systemd-run --user --scope`, `BindsTo=ai-mux.service`): stopping
ai-mux stops every service, `down` stops the scope and runs `post_stop`, only a process in the scope may
register. A dev run outside systemd uses process groups (`AI_MUX_SUPERVISOR=systemd|plain`, `AI_MUX_UNIT`).

## CLI (finds the daemon through `-f FILE`'s `listen`, default the file above, or `AI_MUX_URL`)

    ai-mux serve [-f FILE]            the daemon
    ai-mux ps                         every service: kind, state, pid, last run or job progress
    ai-mux up NAME | down NAME        desired state; rewrites `up:` (`up` re-runs a done job)
    ai-mux events [-since TS]         recorded events since TS (RFC 3339 or Unix seconds), then live

## Logs and events

`~/.local/state/ai-mux/<name>.log`: the service's stdout and stderr plus ai-mux lines for each start, stop and
exit code (`log` in `GET /v1/services`). `~/.local/state/ai-mux/events.db`: SQLite `events(ts, service, run,
type, json)`, every event. ai-mux's own log: `journalctl --user -u ai-mux`.
