# Running ai-mux

ai-mux (PROTOCOL.md) is the one process on a host that serves AI: it starts every endpoint's engine,
takes its registration on a Unix socket and serves the Responses API over HTTP.

    cd codex && go build -o bin/ai-mux ./cmd/ai-mux && go test ./...

## Compose file: `~/.config/ai-mux/ai-mux.yaml` (an example: hosts/sentry.yaml)

`listen`, `socket`, and per endpoint: `up`, `gguf` (tokenizer + chat template), `ctx`, `efforts`, `vision`,
`device`, `command`, `cwd`, `env`, `post_stop`, `restart` (`on-failure`: 10 s, 20 s, 40 s … 5 min; or `no`).
The endpoint process gets `AI_MUX_SOCK` and `AI_MUX_ENDPOINT`.

## Service

    install -m755 bin/ai-mux ~/.local/bin/ai-mux
    install -Dm644 ai-mux.service ~/.config/systemd/user/ai-mux.service
    systemctl --user daemon-reload && systemctl --user enable --now ai-mux
    loginctl enable-linger $USER          # keep it running when logged out

Each endpoint runs in `ai-mux-<name>.scope` (`systemd-run --user --scope`, `BindsTo=ai-mux.service`): stopping
or killing ai-mux stops every endpoint, `down` stops the scope and runs `post_stop`. Only a process in that
scope may register. Outside systemd (a dev run) endpoints are plain process groups killed with ai-mux
(`AI_MUX_SUPERVISOR=systemd|plain` forces one; `AI_MUX_UNIT` names the unit scopes bind to).

## CLI (finds the daemon through `-f FILE`'s `listen`, default the file above, or `AI_MUX_URL`)

    ai-mux serve [-f FILE]     the daemon
    ai-mux ps                  every endpoint: state, pid, ctx, slots, last run's prefill/decode tok/s and idle %
    ai-mux up NAME | down NAME desired state; also rewrites `up:` in the file
    ai-mux events              the /v1/events stream

## Logs

stdout and stderr of each endpoint: `~/.local/state/ai-mux/<name>.log` (appended; ai-mux lines mark every
start, stop and exit code). The path is `log` in `GET /v1/endpoints`; a failed endpoint's `reason` is there too.
ai-mux's own log is the journal: `journalctl --user -u ai-mux`.
