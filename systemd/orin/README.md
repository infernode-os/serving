# systemd/orin/ — user-scope service unit for SGLang on Jetson Orin

These are the artifacts that bring `serving-sglang.service` up on a Jetson Orin AGX, managed by **the user's** systemd manager (not the system one). InferNode's other LLM-stack services on Hephaestus (`ollama.service`, `infernode-llm.service`) run the same way; this slots in as a sibling.

Running as a user unit means:
- No `/etc/` files, no `sudo` to install or manage.
- No polkit rules — `systemctl --user start/stop` works for the owning user without auth.
- Lifecycle is tied to the user session; `loginctl enable-linger <user>` keeps it across logouts (already enabled for `pdfinn` on Hephaestus).

## Files

| File | Goes to | Purpose |
|---|---|---|
| `serving-sglang.service` | `~/.config/systemd/user/serving-sglang.service` | The unit |
| `serving-sglang.env.example` | `~/.config/systemd/user/serving-sglang.env` (with edits) | Runtime config (image pin, model, port, etc.) |

## Install (one-time per host)

```sh
mkdir -p ~/.config/systemd/user
cp systemd/orin/serving-sglang.service ~/.config/systemd/user/
cp systemd/orin/serving-sglang.env.example ~/.config/systemd/user/serving-sglang.env
# edit serving-sglang.env: pin IMAGE to a published :orin-<sha>, pick a model
$EDITOR ~/.config/systemd/user/serving-sglang.env

# Linger lets the user's systemd manager survive logout (so the service
# keeps running on a headless box). One-time, requires sudo. Skip if
# already enabled — verify with `loginctl show-user $USER | grep Linger`.
sudo loginctl enable-linger "$USER"

systemctl --user daemon-reload
```

The unit is intentionally **not enabled** by default — `serving-sglang` is meant to be brought up and down by the `llmctl` switcher tool in `infernode-os/infernode` (so only one local LLM backend runs at a time). For a v1 deploy that doesn't yet have `llmctl`, you can `systemctl --user enable --now serving-sglang.service` to make it start at session login, but be sure Ollama is stopped first.

## Run / stop / status

```sh
systemctl --user start   serving-sglang.service
systemctl --user stop    serving-sglang.service
systemctl --user status  serving-sglang.service
journalctl --user -u serving-sglang.service -f
```

## Verifying after start

```sh
# /health (once available — sglang sets up gradually)
for i in $(seq 1 60); do
  curl -sf -m 2 http://127.0.0.1:30000/v1/models >/dev/null && echo "ready" && break
  sleep 2
done

# Chat completion (note: first call after start triggers flashinfer JIT
# compilation — expect ~50–60s cold, sub-second warm)
curl -sf -m 180 http://127.0.0.1:30000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","messages":[{"role":"user","content":"2+2="}],"max_tokens":16,"temperature":0}'
```

## Hephaestus specifics

The env example sets `DOCKER_HOST=unix:///run/docker-dev.sock` — that's the **dev** daemon (SSD-rooted), required on Hephaestus per the dual-Docker-daemon model documented in `CLAUDE.md` and `runbooks/hephaestus-deploy.md` §0.1. On a field-deployment Orin AGX with a single daemon, set `DOCKER_HOST=unix:///var/run/docker.sock` (or unset the variable and the `docker` CLI uses its default).

## Why `Type=exec` + `/bin/sh -c`?

The ExecStart wraps `docker run` in a small shell so that `$EXTRA_ARGS` (used to inject e.g. `--tokenizer-path` for Llama-3 family) can be empty without passing an empty argv element to argparse. systemd's own `${VAR}` expansion would pass `""` as an argument; shell word-splitting drops it. `exec` keeps PID continuity so `Type=exec` semantics still apply.

## Why `SuccessExitStatus=137 143`?

`docker stop` SIGKILLs the container if it doesn't drain inside the grace period — that surfaces as exit 137 (or 143 for SIGTERM). Treating those as clean exits keeps `systemctl --user is-active` correct after a normal stop. The unit's `ExecStop` uses `docker stop -t 30` so SGLang has time to drain mid-JIT-compilation; 137 is just the belt to the suspenders.
