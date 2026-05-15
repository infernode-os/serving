# sglang/orin/ — Jetson Orin AGX (sm_87)

Container recipe for SGLang on Jetson Orin AGX. **v1 is a thin delta on the spike-proven `dustynv/sglang:r36.4.0` base** — the same bundle that served TinyLlama and produced the ~3× concurrent-throughput win over Ollama in the 2026-05-14 bake-off (see `docs/SGLANG-ADOPTION-NOTES.md` §"Spike attempt 2").

## Target

| Property | Value |
|---|---|
| GPU | Ampere Tegra (`sm_87`) |
| JetPack | 6.x (R36.4 series) |
| CUDA (container) | 12.6 |
| L4T base | `r36.4` |
| Python (container) | 3.10 |
| Ubuntu (container) | 22.04 |
| PyTorch | 2.5.0+cu126 (sm_87 + `USE_DISTRIBUTED=1`, from dustynv) |
| Triton | 3.2.0 |
| SGLang | **0.4.1.post7** (dustynv-bundled, editable at `/workspace/sglang`) |
| sgl-kernel | 0.0.2.post15 (sm_87 binaries) |

## Why this exact base

The dustynv `r36.4.0` tag is currently the only public SGLang container in which **all four of** `torch built with USE_DISTRIBUTED=1`, **sm_87 device code**, **a working sgl-kernel for sm_87**, and **a triton matching that torch** coexist. INFR-68's spike attempt 1 ruled out the official PyTorch wheels (cu126 wheel has no sm_87; NVIDIA's JP6 torch has no `USE_DISTRIBUTED`); INFR-91/92 ruled out a delta upgrade onto `r36.4-cu129-24.04` (sglang 0.5.x strictly pins an sgl-kernel that ships only sm_90/sm_100 binaries; cross-arch fallback fails at first kernel call). v1 ships what actually serves a request, with v0.5.x deferred to INFR-92.

Trade-offs accepted at this pin:

- **No `gpt_oss.py` model arch** (added Aug 2025; this SGLang is Feb 2025). lucibridge's gpt-oss routing falls back to Ollama per `runbooks/lucibridge-routing.md`.
- **No MXFP4 quantisation** in 0.4.1's supported list. Re-quantise gpt-oss to AWQ/GPTQ/FP8 if you need it on this image; otherwise wait on INFR-92.

Everything else lucibridge routes through SGLang (dispatch / tool_call / memory / task, with non-gpt-oss models) works.

## Build-time deltas applied on top of the base

The Dockerfile is small on purpose. Three changes:

1. **Remove the dataclasses backport** at `/usr/local/lib/python3.10/dist-packages/dataclasses.py`. dustynv ships the Python-2.7 `dataclasses-0.x` package, which shadows the Python 3.10 stdlib module and breaks any modern decorator-based dataclass code (most directly: `torch._inductor.runtime.hints` calls `fields(AttrsDescriptor)` at module load). Spike fix carried over.
2. **Bake `TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1`** as image ENV. Container's Triton 3.2 + Torch 2.5 combo breaks `torch._inductor` at import. We don't need `torch.compile` for serving. Spike note carried over.
3. **Bake Llama-3 / Llama-3.1 tokenizer dirs at `/opt/tokenizers/`** (INFR-78). SGLang's GGUF tokenizer doesn't register Llama-3 special tokens correctly; pointing `--tokenizer-path` at a real HuggingFace tokenizer dir at launch time is the fix.

That's the whole delta. The base image is otherwise untouched.

## Build (CI, GitHub-hosted)

CI builds on `ubuntu-24.04-arm` (Graviton SBSA, native aarch64 — no QEMU). See `.github/workflows/build-sglang.yml`. The Dockerfile carries three build-time guards that fail the build if torch loses CUDA, if Triton or SGLang fail to import, or if the SGLang version pin drifts.

## Build (manual on Hephaestus)

```sh
cd ~/serving/sglang/orin
docker --host unix:///run/docker-dev.sock build \
  --build-arg BASE_IMAGE=dustynv/sglang:r36.4.0 \
  -t serving-sglang:orin-local .
```

(Hephaestus-specific: use the dev daemon socket via `--host` per the dual-daemon policy in `runbooks/hephaestus-deploy.md` §0.1. On a field-deployment Orin AGX, drop the `--host` flag.)

## Files

| File | Origin | Purpose |
|---|---|---|
| `Dockerfile` | InferNode-authored | Thin delta on `dustynv/sglang:r36.4.0` |
| `validate-on-hardware.sh` | InferNode-authored | On-hardware validator with real `/v1/chat/completions` smoke + KV-cache assertion |
| `bake-tokenizers.sh` | InferNode-authored | Downloads Llama-3 / Llama-3.1 tokenizer dirs into `/opt/tokenizers/` (INFR-78) |
| `test.py` | verbatim from `dusty-nv/jetson-containers` | Basic smoke test (`import sglang`, print version + CUDA device) |
| `Dockerfile.upstream` | verbatim from `dusty-nv/jetson-containers` | **Unused.** Reference copy of the framework's install path (ruled out in INFR-91). Kept for diff against upstream re-syncs. |
| `config.py` | from `dusty-nv/jetson-containers`, **was modified** | **Unused.** jetson-containers package config; no longer driven by our build. |
| `install.sh` / `build.sh` | verbatim from `dusty-nv/jetson-containers` | **Unused.** Drove the previous framework build path; INFR-91 documents why it didn't produce a working image standalone. |

The "unused" files are retained for now under MIT attribution (`../LICENSE-UPSTREAM.md`). They can be deleted in a future cleanup if no one revisits the framework path within ~1 release cycle.

## What this does NOT include

- **Entrypoint scripts.** Launch arguments live with the runbook (`runbooks/hephaestus-deploy.md`) rather than baked into the image, so the same image serves different model paths without rebuild.

## Verifying a build (on hardware)

The validator runs all seven checks including a real serving smoke (launches `sglang.launch_server` with TinyLlama, asserts `/health`, asserts the `KV Cache is allocated` startup line, exercises `/v1/chat/completions`):

```sh
# On Hephaestus dev daemon (mount the host HF cache so TinyLlama isn't
# redownloaded each run):
docker --host unix:///run/docker-dev.sock run --rm \
  --runtime nvidia --gpus all \
  -v /mnt/orin-ssd/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  ghcr.io/infernode-os/serving-sglang:orin-latest \
  /opt/sglang/validate-on-hardware.sh
```

On success the final line is `All on-hardware checks passed.`. Expected serving-smoke output:

```
=== 7. Serving smoke (TinyLlama → /v1/chat/completions, KV-cache verified) ===
  /health up after ~55s
  KV cache log line: KV Cache is allocated. K size: ~8.8 GB, V size: ~8.8 GB.
  ok: KV cache allocated
  POST /v1/chat/completions
  completion: 'The answer to the question is 4.'
```

`gpt-oss` arch verification — deferred to INFR-92, expected to fail on v1:

```sh
docker --host unix:///run/docker-dev.sock run --rm --runtime nvidia --gpus all \
  ghcr.io/infernode-os/serving-sglang:orin-latest \
  python3 -c "import sglang.srt.models.gpt_oss" || echo "(expected for v1; INFR-92)"
```
