# Hephaestus deploy runbook — SGLang on Jetson Orin AGX

End-to-end runbook for deploying the GHCR-built SGLang container on
**Hephaestus** (Jetson Orin AGX, JetPack 6.x, 64 GiB unified memory)
and wiring it into the existing `serve-llm.sh` / `lucibridge`
operational model.

**Owning epic:** INFR-73. **Acceptance gate:** a new contributor can
bring SGLang up on Hephaestus in under 15 minutes following this
document.

---

## 0. Prereqs

This runbook covers two deployment shapes:

* **Field deployment** — a vanilla Jetson Orin AGX with a single disk
  and a single Docker daemon (the standard system one). This is the
  intended end-user path. Most readers should follow this.
* **Hephaestus (dual-purpose dev box)** — both a field-parity reference
  *and* a development environment. Has a 916 GiB `/mnt/orin-ssd` in
  addition to the root partition, and runs a **second** Docker daemon
  (`docker-dev.service`) on the SSD specifically for experimental work
  (SGLang, anything else that would otherwise crush root). See §0.1.

### Hardware + driver prereqs (both shapes)

```sh
cat /etc/nv_tegra_release | head -1            # expect R36, REVISION: 4.x
nvidia-smi                                      # expect Orin / CUDA 12.6+
docker info | grep -iE 'server version|runtimes'  # expect 24+ and nvidia runtime registered
```

The `nvidia` runtime must be registered with whichever Docker daemon
will run SGLang. Verify with `docker info --format '{{.Runtimes}}'`
— output should include `nvidia`. If missing, install
`nvidia-container-toolkit` and re-add `"runtimes": {"nvidia": ...}`
to that daemon's `daemon.json`.

### Field deployment (single-disk Orin AGX)

The standard system Docker daemon's storage on `/var/lib/docker`. The
~12-14 GiB SGLang image + runtime data all live there. Verify space:

```sh
df -h /                                        # need ≥20 GiB free for image + working set
```

Skip §0.1 entirely; jump to §1.

### Hephaestus disk policy (load-bearing — do NOT violate on this host)

Hephaestus serves two purposes simultaneously: **(a)** field-parity
reference for TAK / NERVA / Ollama (which must stay on root partition,
exactly as they would on a no-SSD field unit), and **(b)** development
environment for experimental work. Because a single Docker daemon has
exactly one storage root, the only way to satisfy both is **two
daemons**.

| Daemon | Socket | Storage | Used by |
|---|---|---|---|
| `docker.service` (production) | `/run/docker.sock` | root partition (`/var/lib/docker` symlinked to `/mnt/orin-ssd/docker/docker`, but containerd content store at `/var/lib/containerd` lives on root — both halves end up landing pulled images on root via the shared system containerd) | TAK · NERVA · Ollama · anything mirroring field |
| `docker-dev.service` (experimental, see INFR-90) | `/run/docker-dev.sock` | `/mnt/orin-ssd/docker-dev` (own legacy snapshotter, `containerd-snapshotter: false`, completely off the shared containerd path) | SGLang · any other Jetson experiment |

**Do not** migrate TAK / NERVA / Ollama to the dev daemon — they belong
on production for field parity. **Do not** install SGLang on the
production daemon — its 12-14 GiB image plus working set would crush the
intentionally-constrained root partition. Use `docker-dev.service` for
all of §1-§7 below.

See §0.1 for one-time dev-daemon setup if it isn't running yet.

---

## 0.1 Dev-daemon setup (Hephaestus only — one-time, INFR-90)

Skip this entire section on a field-deployment Orin AGX.

If `systemctl is-active docker-dev` returns `active` and
`DOCKER_HOST=unix:///run/docker-dev.sock docker info` succeeds, the
dev daemon is already up — skip ahead to §1.

Otherwise, set it up:

1. Create `/etc/docker/daemon-dev.json`:

   ```json
   {
     "data-root": "/mnt/orin-ssd/docker-dev",
     "exec-root": "/var/run/docker-dev",
     "pidfile": "/run/docker-dev.pid",
     "hosts": ["unix:///run/docker-dev.sock"],
     "bridge": "docker1",
     "default-address-pools": [{ "base": "172.31.0.0/16", "size": 24 }],
     "features": { "containerd-snapshotter": false },
     "runtimes": {
       "nvidia": { "args": [], "path": "nvidia-container-runtime" }
     },
     "log-driver": "json-file",
     "log-opts": { "max-size": "100m", "max-file": "3" }
   }
   ```

   `containerd-snapshotter: false` is critical — it puts image content
   under `data-root` instead of the shared `/var/lib/containerd` on
   root. Without this, image pulls on the dev daemon would still land
   on root via the shared containerd, defeating the whole point.

2. Create `/etc/systemd/system/docker-dev.service`:

   ```ini
   [Unit]
   Description=Docker Application Container Engine (dev / SSD-rooted)
   Documentation=https://docs.docker.com
   RequiresMountsFor=/mnt/orin-ssd
   After=network-online.target nss-lookup.target containerd.service docker.service
   Wants=network-online.target containerd.service
   StartLimitBurst=3
   StartLimitIntervalSec=60

   [Service]
   Type=notify
   ExecStart=/usr/bin/dockerd --config-file=/etc/docker/daemon-dev.json
   ExecReload=/bin/kill -s HUP $MAINPID
   TimeoutStartSec=0
   TimeoutStopSec=120s
   RestartSec=2
   Restart=always
   LimitNOFILE=infinity
   LimitNPROC=infinity
   LimitCORE=infinity
   TasksMax=infinity
   Delegate=yes
   KillMode=process
   OOMScoreAdjust=-500

   [Install]
   WantedBy=multi-user.target
   ```

3. Add a drop-in for the bridge — Docker refuses to start with a
   non-default bridge name unless that bridge already exists in the
   kernel, and kernel bridges don't persist across reboots:

   ```sh
   sudo mkdir -p /etc/systemd/system/docker-dev.service.d
   sudo tee /etc/systemd/system/docker-dev.service.d/bridge.conf <<'EOF'
   [Service]
   ExecStartPre=/bin/sh -c 'ip link show docker1 >/dev/null 2>&1 || ip link add docker1 type bridge'
   ExecStartPre=/bin/sh -c 'ip link set docker1 up'
   EOF
   ```

4. Enable + start:

   ```sh
   sudo systemctl daemon-reload
   sudo systemctl enable --now docker-dev.service
   ```

5. Add a shell helper so you don't have to remember the socket path:

   ```sh
   echo "alias dev-docker='docker --host unix:///run/docker-dev.sock'" >> ~/.bashrc
   source ~/.bashrc
   ```

6. Verify:

   ```sh
   dev-docker info --format '{{.ServerVersion}} | {{.DockerRootDir}} | runtimes: {{.Runtimes}}'
   # Expect: 29.x | /mnt/orin-ssd/docker-dev | runtimes: ... nvidia ...
   ```

The bridge subnet `172.31.0.0/16` was picked to avoid colliding with
prod's `docker0` (172.17), TAK (172.18), NERVA (172.19), TBL4
(172.20-21), ZeroTier (10.243), and LAN (192.168.1). If you have other
networks on the host, audit with `ip route` and pick a free /16.

---

## 1. Pull the container

GHCR images are public; no docker login required.

**On Hephaestus**, prefix every `docker` command in this section
through to §7 with `dev-docker` (or `docker --host
unix:///run/docker-dev.sock`) so it hits the dev daemon, not the
production one. **On a field-deployment Orin AGX**, use plain
`docker`.

For convenience the snippets below use plain `docker`; mentally
substitute `dev-docker` if you're on Hephaestus.

```sh
# For end-user trials and dev: use :orin-latest (always points at the
# tip of main).
IMAGE='ghcr.io/infernode-os/serving-sglang:orin-latest'

# For production/field deploys: pin a specific short-SHA so the version
# is what gets recorded in incident timelines.
# IMAGE='ghcr.io/infernode-os/serving-sglang:orin-<short-sha>'

docker pull "$IMAGE"
docker images "$IMAGE" --format '{{.Repository}}:{{.Tag}} {{.Size}}'
# Expect: ~12–14 GB (CUDA + cuDNN + PyTorch wheels + SGLang + sgl-kernel).
# Verify residual headroom — on field deploy this is on root partition;
# on Hephaestus this is on /mnt/orin-ssd via the dev daemon.
df -h /
df -h /mnt/orin-ssd  # Hephaestus only
```

If the root partition is tight, `docker image prune` old SGLang images
**only** (`docker images "ghcr.io/infernode-os/serving-sglang" --quiet | tail -n +3`)
— do not prune TAK/NERVA images.

---

## 2. Pre-flight checks

Run the on-hardware validator. It checks CUDA + sglang import + sgl_kernel
arch + launch_server entrypoint + tokenizer bake, **and** runs a real
TinyLlama serve through `/v1/chat/completions`, asserting the `KV Cache is
allocated` startup line appears.

```sh
HF_CACHE=/mnt/orin-ssd/huggingface          # Hephaestus dev daemon
# HF_CACHE=/var/lib/huggingface             # field-deployment Orin AGX
mkdir -p "$HF_CACHE"

docker run --rm \
  --runtime nvidia --gpus all \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  "$IMAGE" \
  /opt/sglang/validate-on-hardware.sh
```

Expected last line: `All on-hardware checks passed.`. The serving smoke
takes ~60s on a cold cache (TinyLlama download + load + 8-token completion);
re-runs reuse the cache and complete in ~10–15s.

If anything fails, **stop**. Don't proceed to §3. The image is broken;
either pull a different tag or rebuild from `sglang/orin/` (see
`sglang/orin/README.md`).

> **v1 known skip** — step 3 (`gpt_oss arch importable`) is informational
> and prints "skip: gpt_oss not present in this SGLang version". v1 ships
> SGLang 0.4.1.post7 which predates gpt-oss; INFR-92 covers the upgrade.
> lucibridge's routing falls back to Ollama for gpt-oss in the meantime.

---

## 3. Launch (one-shot, for testing)

The launch invocation that produced the bake-off in
`docs/SGLANG-ADOPTION-NOTES.md`, adapted for the GHCR image. Bind-mounts
keep the HF cache on `/mnt/orin-ssd` (disk policy §0).

**TinyLlama smoke** — the same payload the validator runs, but interactive
so you can watch the logs and curl by hand:

```sh
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0       # smoke target — uses Llama-2 tokenizer; no --tokenizer-path needed
HF_CACHE=/mnt/orin-ssd/huggingface
LOGS=/mnt/orin-ssd/pdfinn/scratch/sglang-logs
mkdir -p "$HF_CACHE" "$LOGS"

docker run --rm -it \
  --runtime nvidia --gpus all \
  --network host \
  --shm-size 8g \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 127.0.0.1 --port 30000 \
    --attention-backend triton \
    --mem-fraction-static 0.5 \
    --disable-cuda-graph \
    --log-level info \
  2>&1 | tee "$LOGS/sglang-$(date +%s).log"
```

**Llama-3 family launch** — for any Llama-3.x model (e.g. `unsloth/Meta-Llama-3.1-8B-Instruct` or a Llama-3-GGUF blob from Ollama's store), add the tokenizer override:

```
    --tokenizer-path /opt/tokenizers/llama-3.1 \
```

Without that override, Llama-3 special tokens (`<|eot_id|>`, `<|begin_of_text|>`, etc.) aren't registered and stop strings won't match — INFR-78. The validator's TinyLlama smoke does *not* need it.

Flag notes (all carried forward from spike findings):

* `--runtime nvidia --gpus all` — required for Tegra GPU passthrough.
* `--network host` — simplest port binding; firewall lives on the host.
  Skip `--publish` to avoid double NAT through Docker.
* `--shm-size 8g` — SGLang's worker pool uses shared memory; default
  64 MB causes silent stalls under concurrency.
* `--attention-backend triton` — most conservative Jetson path
  (flashinfer also works but Triton was the proven default in the spike).
* `--mem-fraction-static 0.5` — leaves 32 GiB on the unified-memory
  budget for OS + Ollama + headroom. Tune per workload (see §6).
* `--disable-cuda-graph` — CUDA graph capture is finicky on Jetson;
  disable for stability. Re-enable later if perf needs it.
* `TORCHDYNAMO_DISABLE=1 / TORCH_COMPILE_DISABLE=1` — bypass
  `torch.compile`; the in-image Triton + torch combination doesn't
  reliably JIT-compile and we don't need compile for serving. **Already
  baked into the image as ENV** — pass on the command line only to
  override.

---

## 4. Healthcheck sequence

Run this against a freshly-launched server, in order:

```sh
BASE=http://127.0.0.1:30000

# 1. liveness — must return 200 immediately
curl -fsS "$BASE/health"

# 2. model list — must include the --model-path you launched with
curl -fsS "$BASE/v1/models" | python3 -m json.tool

# 3. smoke chat completion — should complete in <2s for TinyLlama
curl -fsS "$BASE/v1/chat/completions" -H 'content-type: application/json' -d '{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "messages": [{"role":"user","content":"Capital of Belgium? One word."}],
  "max_tokens": 8,
  "temperature": 0
}'
# Expect: "Brussels" or close. If you see garbage tokens, the tokenizer
# path is wrong — see §7 troubleshooting.
```

---

## 5. systemd unit (user-scope)

The unit is **user-scope** — runs under the user's `systemctl --user` manager alongside `ollama.service` and `infernode-llm.service`. No `/etc/`, no sudo, no polkit. Canonical artifacts live in `systemd/orin/` in this repo (see `systemd/orin/README.md`).

Install:

```sh
cd ~/github.com/infernode-os/serving        # or wherever you checked out
mkdir -p ~/.config/systemd/user
cp systemd/orin/serving-sglang.service     ~/.config/systemd/user/
cp systemd/orin/serving-sglang.env.example ~/.config/systemd/user/serving-sglang.env

# Pin IMAGE to a published :orin-<sha>; pick a model. The example ships
# with TinyLlama as a smoke target — change MODEL_PATH (and uncomment
# EXTRA_ARGS for Llama-3 family) for production.
$EDITOR ~/.config/systemd/user/serving-sglang.env

# One-time: enable lingering so the user manager survives logout on a
# headless box. Verify first — Hephaestus has this enabled already.
loginctl show-user "$USER" | grep -q 'Linger=yes' || sudo loginctl enable-linger "$USER"

systemctl --user daemon-reload
```

The unit is **not enabled by default** — bring it up via the InferNode `llmctl` switcher (in `infernode-os/infernode`) so only one local backend runs at a time. For a deploy that pre-dates `llmctl`, you can `systemctl --user enable --now serving-sglang.service`, but you are then responsible for stopping `ollama.service` before starting it.

Run / stop:

```sh
systemctl --user start  serving-sglang.service
systemctl --user stop   serving-sglang.service
systemctl --user status serving-sglang.service --no-pager
journalctl --user -u serving-sglang.service -f
```

The env file (`serving-sglang.env.example` → installed copy) controls everything tunable — image pin, model, port, `mem-fraction-static`, HF cache mount, extra launch flags. See `systemd/orin/README.md` for the field-by-field meanings and the Hephaestus-vs-field-deployment differences (notably `DOCKER_HOST=unix:///run/docker-dev.sock` for Hephaestus's dev daemon, default socket for a field Orin AGX).

> **First-call latency** — flashinfer JIT compiles its kernels on the first `/v1/chat/completions` call (≈50–60s on Orin). Subsequent calls are sub-second. Build the JIT cache once after each start by issuing one warm-up call before any client traffic. Future work: mount a persistent JIT cache to skip the cold compile.

---

## 6. Memory budgeting on 64 GiB unified

Jetson Orin's 64 GiB is shared between CPU and GPU. Concrete budget
for the v1 cut, drawn from the spike's measured working set:

| Component | Resident (typical) | Notes |
|---|---|---|
| OS + drivers | ~3 GiB | baseline |
| Ollama + one resident model (Devstral GGUF Q4) | ~14 GiB | `OLLAMA_KEEP_ALIVE` default |
| SGLang model weights (Llama-3.1-8B Q4 GGUF, v1) | ~5 GiB | bake-off shape; INFR-92 swaps in gpt-oss-20b at ~13 GiB |
| SGLang KV cache | ~9 GiB at 4096 ctx, 16 running | `--mem-fraction-static 0.5` cap; the validator's TinyLlama smoke logs `K=8.8 GB V=8.8 GB` |
| Headroom | ~25–33 GiB | TAK / NERVA / NN bursts |

Tunables:

* **`--mem-fraction-static`** = fraction of GPU memory SGLang
  pre-allocates for weights + activations. `0.5` is the spike default
  (32 GiB cap). Raise to `0.6` if Ollama is dropped from the box;
  drop to `0.4` if TAK/NERVA models grow.
* **`--max-running-requests`** = concurrency cap. `16` was the spike
  default; tail latency blew out at N=16 due to starvation. For
  Veltro's expected fan-out (≤8 concurrent), set `--max-running-requests 8`.
* **`--max-total-tokens`** = global token budget across batch. Keep
  proportional to `mem-fraction-static * gpu_mem`.

After any tune, re-run §4 healthchecks and validate p95 latency in
`docs/SGLANG-ADOPTION-NOTES.md`'s bake-off shape.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA unknown error` at container start | Tegra driver shim not on `LD_LIBRARY_PATH` inside container | Verify `--runtime nvidia` is set; the runtime injects `/usr/lib/aarch64-linux-gnu/tegra` automatically |
| Garbage output, off-topic answers (Belgium → "It seems like you are referring to…") | GGUF tokenizer used for Llama-3 family (INFR-78 known issue) | Set `--tokenizer-path /opt/tokenizers/llama-3.1` |
| `stop` strings never match, completions run to `max_tokens` | Same as above — special tokens not registered | Same as above |
| OOM during model load | Other process holds GPU memory | `nvidia-smi` to find culprit; drop Ollama's resident model with `curl -X POST :11434/api/generate -d '{"model":"...","keep_alive":0}'` |
| Server starts but `/health` 503s | Worker pool stalled on small shm | Confirm `--shm-size 8g` is on the `docker run` line |
| `import sglang` fails inside container | Stale image / partial install | Re-pull pinned tag; re-run §2 pre-flight |
| Per-request latency >2× expected | `torch.compile` accidentally enabled | Confirm `TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1` |
| Build/push CI green but pull on Hephaestus 404s | GHCR visibility set wrong on first publish | One-time: in repo Settings → Packages, set the package public |

---

## 8. `serve-llm.sh` integration

`serve-llm.sh` (in `infernode-os/infernode`) is the dev-side launcher
that the ZeroTier-mounted user interacts with. Today it talks to
Ollama at `http://127.0.0.1:11434/v1`. With SGLang available the
launcher gains a sibling backend.

### Single-backend mode (Ollama-only — current default)

No change. SGLang need not be installed.

### Dual-backend mode (Ollama + SGLang — new)

Set these in the env of `serve-llm.service` (or before invoking
`serve-llm.sh` interactively):

```sh
export LLM_BACKEND_DEFAULT=http://127.0.0.1:11434/v1
export LLM_BACKEND_SGLANG=http://127.0.0.1:30000/v1
```

`lucibridge` (see §9) picks per-tool which URL to dispatch to. If
`LLM_BACKEND_SGLANG` is unset, lucibridge falls back to
`LLM_BACKEND_DEFAULT` for every request — backward compatible.

### Switching modes

Once `llmctl` lands in `infernode-os/infernode`, the right way to switch is `llmctl set ollama` / `llmctl set sglang` — it stops one cleanly and starts the other, then updates `/lib/ndb/llm` accordingly. Until then, manual:

```sh
# Ollama-only — stop SGLang first, then start Ollama
systemctl --user stop  serving-sglang.service
systemctl --user start ollama.service

# SGLang-only — opposite direction
systemctl --user stop  ollama.service
systemctl --user start serving-sglang.service
```

Both units are user-scope; no sudo. They are intentionally not `Conflicts=` of each other in the unit files — `llmctl` (or you, manually) owns the stop-before-start ordering.

---

## 9. lucibridge per-tool routing (cross-ref INFR-79)

See `runbooks/lucibridge-routing.md` for the routing config schema and
the per-tool / per-capability mapping. Headline:

* `tool_category in {limbo_authoring}` → `LLM_BACKEND_DEFAULT`
  (Devstral via Ollama, current production)
* `tool_category in {dispatch, tool_call, memory, task}` →
  `LLM_BACKEND_SGLANG` (gpt-oss via SGLang, post-INFR-77)
* unset / unknown category → `LLM_BACKEND_DEFAULT` (fallback)

The routing change lives in `infernode-os/infernode`'s `lucibridge`
module; this runbook is the operational doc that documents what the
deployed config looks like and how to flip between modes.

---

## 10. Stopping cleanly

```sh
# Graceful — gives SGLang up to 30s to drain in-flight requests
# (ExecStop in the unit is `docker stop -t 30`).
systemctl --user stop serving-sglang.service

# Ad-hoc (skips the unit, useful if systemd is unhappy)
docker --host unix:///run/docker-dev.sock stop -t 30 serving-sglang

# If even that hangs (rare; usually means a streaming client is pinning
# the connection), escalate
docker --host unix:///run/docker-dev.sock kill --signal=KILL serving-sglang
```

Expected drain time is sub-second when idle, up to ~30s under N≥16 concurrent. If `docker stop` takes longer than 60s, it usually means a downstream client is holding a streaming request open; the bridge should be killed first (`systemctl --user stop infernode-llm.service`).

---

## 11. Verifying end-to-end with `infernode-llm` + lucibridge

Single-backend (v1 default — `llmctl` not yet implemented):

```sh
# Pick a backend; stop the other.
systemctl --user stop ollama.service
systemctl --user start serving-sglang.service
sleep 60                                       # JIT warm-up, model load
curl -sf http://127.0.0.1:30000/v1/models      # confirm SGLang up

# The infernode-llm 9P gateway already runs (user unit). It reads
# /lib/ndb/llm inside emu for backend URL — make sure that points at
# http://127.0.0.1:30000/v1 when SGLang is the active backend.
systemctl --user status infernode-llm.service --no-pager | head -5
```

A passing run looks like: SGLang's journal shows one `POST /v1/chat/completions` per request from the bridge; the bridge's logs show the chosen URL and tokens flowing.

When the lucibridge per-tool routing (INFR-79) and `llmctl` switcher land, this section becomes:
- `llmctl set sglang` (or `ollama`, or pick a mix per tool category)
- single Veltro-shaped probe that exercises both backends via the routing
- journal evidence on each side

---

## References

* `docs/SGLANG-ADOPTION-NOTES.md` — spike findings, bake-off numbers, the original launch flags
* `sglang/orin/README.md` — container build / version-pin details
* `runbooks/lucibridge-routing.md` — routing config schema (INFR-79)
* `infernode-os/infernode:docs/HEADLESS-LLM-DAEMON.md` — `serve-llm.service` operational template
* Tickets: INFR-73 (epic), INFR-77 (gpt-oss unblock), INFR-78 (tokenizer), INFR-79 (routing), INFR-80 (this runbook)
