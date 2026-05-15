# Deploy runbook — SGLang on Jetson Orin AGX

End-to-end runbook for deploying the GHCR-built SGLang container on
a Jetson Orin AGX (JetPack 6.x, 64 GiB unified memory) and wiring it
into the `serve-llm.sh` / `lucibridge` operational model.

**Owning epic:** INFR-73. **Acceptance gate:** a new contributor can
bring SGLang up on a clean Orin AGX in under 15 minutes following this
document.

---

## 0. Prereqs

This runbook covers the **field-deployment shape**: a vanilla Jetson
Orin AGX with a single disk and a single Docker daemon (the standard
system one).

### Hardware + driver prereqs

```sh
cat /etc/nv_tegra_release | head -1            # expect R36, REVISION: 4.x
nvidia-smi                                      # expect Orin / CUDA 12.6+
docker info | grep -iE 'server version|runtimes'  # expect 24+ and nvidia runtime registered
```

The `nvidia` runtime must be registered with the Docker daemon that
will run SGLang. Verify with `docker info --format '{{.Runtimes}}'`
— output should include `nvidia`. If missing, install
`nvidia-container-toolkit` and re-add `"runtimes": {"nvidia": ...}`
to the daemon's `daemon.json`.

### Disk space

The standard system Docker daemon stores images at `/var/lib/docker`.
The ~12-14 GiB SGLang image + runtime data all live there. Verify
space:

```sh
df -h /                                        # need ≥20 GiB free for image + working set
```

---

## 1. Pull the container

GHCR images are public; no docker login required.

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

# Verify residual headroom on the disk that backs Docker's data-root.
df -h /
```

If the disk is tight, `docker image prune` old SGLang images **only**
(`docker images "ghcr.io/infernode-os/serving-sglang" --quiet | tail -n +3`)
— do not prune unrelated workload images that share the daemon.

---

## 2. Pre-flight checks

Run the on-hardware validator. It checks CUDA + sglang import + sgl_kernel
arch + launch_server entrypoint + tokenizer bake, **and** runs a real
TinyLlama serve through `/v1/chat/completions`, asserting the `KV Cache is
allocated` startup line appears.

```sh
HF_CACHE="${HF_CACHE:-/var/lib/huggingface}"
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
`docs/SGLANG-ADOPTION-NOTES.md`, adapted for the GHCR image.

**TinyLlama smoke** — the same payload the validator runs, but interactive
so you can watch the logs and curl by hand:

```sh
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0       # smoke target — uses Llama-2 tokenizer; no --tokenizer-path needed
HF_CACHE="${HF_CACHE:-/var/lib/huggingface}"
LOGS="${LOGS:-/var/log/sglang}"
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

**Llama-3 family launch** — for any Llama-3.x model (e.g. `unsloth/Meta-Llama-3.1-8B-Instruct`), add the tokenizer override:

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

## 5. systemd unit (production)

For a deploy that survives reboot, drop the unit below in
`/etc/systemd/system/serving-sglang.service`. Mirrors the pattern in
IOL's `docs/HEADLESS-LLM-DAEMON.md` for `serve-llm.service`.

```ini
[Unit]
Description=InferNode SGLang serving (Jetson Orin)
After=docker.service ollama.service network-online.target
Wants=docker.service network-online.target

[Service]
Type=exec
Restart=on-failure
RestartSec=10
TimeoutStopSec=60

# Resolved env file holds IMAGE pin, model path, port, etc.
EnvironmentFile=/etc/serving-sglang.env

ExecStartPre=-/usr/bin/docker stop serving-sglang
ExecStartPre=-/usr/bin/docker rm   serving-sglang

ExecStart=/usr/bin/docker run --name serving-sglang --rm \
  --runtime nvidia --gpus all \
  --network host --shm-size 8g \
  -v ${HF_CACHE}:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  ${IMAGE} \
  python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tokenizer-path ${TOKENIZER_PATH} \
    --host 127.0.0.1 --port ${PORT} \
    --attention-backend triton \
    --mem-fraction-static ${MEM_FRACTION_STATIC} \
    --disable-cuda-graph \
    --log-level info

ExecStop=/usr/bin/docker stop serving-sglang

[Install]
WantedBy=multi-user.target
```

Companion `/etc/serving-sglang.env` (chmod 0644, no secrets). The v1 example below serves Llama-3.1-8B; once INFR-92 lands and an SGLang with `gpt_oss.py` ships, swap to `openai/gpt-oss-20b` (and drop `TOKENIZER_PATH` — gpt-oss is not Llama-family):

```sh
IMAGE=ghcr.io/infernode-os/serving-sglang:orin-<short-sha>
MODEL_PATH=unsloth/Meta-Llama-3.1-8B-Instruct
TOKENIZER_PATH=/opt/tokenizers/llama-3.1
PORT=30000
MEM_FRACTION_STATIC=0.5
HF_CACHE=/var/lib/huggingface
```

Activate:

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now serving-sglang.service
sudo systemctl status serving-sglang.service --no-pager
journalctl -u serving-sglang.service -f
```

`After=ollama.service` lets Ollama come up first (port 11434), then
SGLang on 30000. Both coexist on 64 GiB unified memory; see §6.

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
| Headroom | ~25–33 GiB | other resident workloads, bursts |

Tunables:

* **`--mem-fraction-static`** = fraction of GPU memory SGLang
  pre-allocates for weights + activations. `0.5` is the spike default
  (32 GiB cap). Raise to `0.6` if Ollama is dropped from the box;
  drop to `0.4` if other resident models grow.
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
| Build/push CI green but `docker pull` 404s | GHCR visibility set wrong on first publish | One-time: in repo Settings → Packages, set the package public |

---

## 8. `serve-llm.sh` integration

`serve-llm.sh` (in `infernode-os/infernode`) is the dev-side launcher
that talks to local LLM backends. Today it talks to Ollama at
`http://127.0.0.1:11434/v1`. With SGLang available the launcher
gains a sibling backend.

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

```sh
# Ollama-only
sudo systemctl stop serving-sglang
sudo systemctl mask serving-sglang   # prevent restart on reboot

# Re-enable SGLang
sudo systemctl unmask serving-sglang
sudo systemctl start serving-sglang
```

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
# Graceful — gives SGLang ~30s to drain in-flight requests
sudo systemctl stop serving-sglang
# or, ad-hoc:
docker stop serving-sglang

# If the daemon is stuck (>60s), escalate
docker kill --signal=KILL serving-sglang
```

Expected drain time is sub-second when idle, up to ~30s under N≥16
concurrent. If `docker stop` takes longer than 60s, it usually means
a downstream client is holding a streaming request open; the bridge
should be killed first (`systemctl stop serve-llm`).

---

## 11. Verifying end-to-end with `serve-llm` + lucibridge

```sh
# Bring everything up
sudo systemctl start ollama serving-sglang serve-llm
sleep 5

# Healthcheck the bridge endpoint
curl -fsS http://127.0.0.1:8080/health    # serve-llm

# Run a Veltro-shaped probe: a tool-call turn that routes to SGLang
# (gpt-oss) and a Limbo-authoring turn that routes to Ollama (Devstral).
# See runbooks/lucibridge-routing.md for the probe payload.
```

A passing run looks like: SGLang's journal shows one `POST
/v1/chat/completions` per dispatched tool call; Ollama's logs show one
generate for the Limbo authoring turn; bridge logs show the routing
decision for each.

---

## References

* `docs/SGLANG-ADOPTION-NOTES.md` — spike findings, bake-off numbers, the original launch flags
* `sglang/orin/README.md` — container build / version-pin details
* `runbooks/lucibridge-routing.md` — routing config schema (INFR-79)
* `infernode-os/infernode:docs/HEADLESS-LLM-DAEMON.md` — `serve-llm.service` operational template
* Tickets: INFR-73 (epic), INFR-77 (gpt-oss unblock), INFR-78 (tokenizer), INFR-79 (routing), INFR-80 (this runbook)
