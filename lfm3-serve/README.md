# lfm3.jl served by mux on archy (Tesla M60 die 3)

`serve.jl` is an engine of mux (`codex/PROTOCOL.md`): it registers through `codex/julia/Mux.jl`, and mux does the chat template, the
tokenizer (`tokenizer.ggml.pre` lfm3 is in `codex/c/tok.c`), the HTTP faces and the live stats. `codex/services/lfm3-1.7b-nate/service.yaml`
starts it; `codex/hosts/archy.yaml` is archy's `~/.config/mux/mux.yaml` (its `compat_listen` 127.0.0.1:18111 is what nm-z.com/chat reads:
`archy-mk=127.0.0.1:18111` in `NMZ_CHAT_EXTRA` of archy's `nmz-chat-gw.service`).

Install on archy: `go build -a -o ~/.local/bin/mux ./cmd/mux` (build elsewhere and copy: archy has no Go; `-a`, as cgo does not see edits to ../c),
`codex/mux.service` to `~/.config/systemd/user/`, then `systemctl --user enable --now mux`. `mux ps` shows the service `up` once the model is
loaded (about 2 minutes).

The weights are in `/dev/shm/lfm3/nate` (nate-epoch1 `model.safetensors` + `config.json` from engi's `~/models/lfm3-1.7b-nate`) and the Julia depot in
`/dev/shm/jdepot`: tmpfs, gone at a reboot (archy's root disk is nearly full).

The older route (flashd in front of `serve.jl --serve SOCK`, units `lfm3-mk-engine` and `lfm3-mk-chat`) is replaced; its unit files are left disabled on archy.
Die 3 is this engine's alone; `nmz-fast-llm` (LFM2) and `stop-judge-embed` were stopped off dies 2 and 3.
The idle % is the share of a run no request of it is in flight; the live prefill rate counts the cached prefix as run.

Decoding: a prompt's rows run as requests (the batcher task owns the card; prompts that wait share one request), its first token is picked on the host, then the chat joins the
decoding chats and the card samples by itself: `chain.jl` queues up to 32 launches of the whole batch back to back (lfm3.jl's SAMPLE node picks each row's next token with the
model's own temperature, top-k and top-p, writes it into the next launch's row and out through a ring in host memory), and the host only watches the ring and streams the tokens.
A prompt arriving, a chat starting or a chat ending sets the chain's abort word: the launches not yet started do nothing, and the batcher runs again with the new set.
Die 2 has a failing band of VRAM above 5.37 GiB: `fence-die2.service` (`~/.local/bin/fence_hold`, the block list of `flash-sass`'s `kern/fences`) holds it, before mux, so nothing is ever given those blocks.
