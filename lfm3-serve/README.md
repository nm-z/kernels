# lfm3.jl served on archy GPU 3 (M60)

`lfm3-mk-engine.service` runs `lfm3.jl --serve` (flashd's engine protocol on /tmp/lfm3mk.sock); `lfm3-mk-chat.service` is flashd in
front of it (127.0.0.1:18643); `archy-mk=127.0.0.1:18643` in `NMZ_CHAT_EXTRA` of archy's `nmz-chat-gw.service` puts it in the
nm-z.com/chat model list. Install: copy both units to `~/.config/systemd/user/`, `systemctl --user daemon-reload`, `enable --now` both.

The weights are in `/dev/shm/lfm3/nate` (the nate-epoch1 model.safetensors + config.json from engi's `~/models/lfm3-1.7b-nate`) and the Julia
depot in `/dev/shm/jdepot`: tmpfs, gone at a reboot (archy's root disk is full).
