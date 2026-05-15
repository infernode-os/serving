# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Container builds, deployment recipes, and routing configuration for the LLM serving stack behind InferNode (`llmsrv` / `lucibridge` / `serve-llm.sh`). It does **not** contain the OS runtime (that lives in `infernode-os/infernode`) or the training pipeline (that lives in `pdfinn/infernode-os-llm`). Lifecycle here tracks JetPack releases / SGLang versions / hardware generations, not model training cycles.

The repo is intentionally **public** (changed 2026-05-14) so CI can use the free `ubuntu-24.04-arm` GitHub-hosted runner minutes. Do not commit secrets, credential-shaped files, internal hostnames, or names of confidential workloads that may colocate with this stack on a dev box. Operational policy that's specific to a particular deployment lives in private docs, not here.

Work is tracked under the **INFR Jira project's "Productize SGLang serving" epic** (parent: INFR-68). When a file references `INFR-NN`, that's a Jira ticket — see ticket commentary for context Claude can't recover from the tree.

## Hardware-target asymmetry (load-bearing)

The `sglang/` subtree has two variants because the two Jetson generations need different base-image strategies:

- **`sglang/orin/`** — Jetson Orin AGX (`sm_87`, JetPack 6.x, CUDA 12.6). v1 is a **thin delta on `dustynv/sglang:r36.4.0`** — the bundle that served TinyLlama and produced the 3× concurrent-throughput win over Ollama in the 2026-05-14 spike (`docs/SGLANG-ADOPTION-NOTES.md` §"Spike attempt 2"). This bundle is currently the only public combo where `torch built with USE_DISTRIBUTED=1` + sm_87 device code + a matching sgl-kernel + a matching triton coexist. INFR-91/92 documents why every other path tried so far failed.
- **`sglang/thor/`** — Jetson Thor (`sm_103`, JetPack 7.x, CUDA 13+). NVIDIA's official `nvcr.io/nvidia/sglang:25.10-py3` targets this hardware, so the Dockerfile is a thin wrapper that adds only InferNode-specific bits. Forward-looking — no Thor box exists yet.

The Orin Dockerfile applies three deltas on the dustynv base: (1) remove the dataclasses backport that shadows the stdlib, (2) bake `TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1` ENV (Triton 3.2 + Torch 2.5 break torch._inductor at import), (3) bake Llama-3 / Llama-3.1 tokenizer dirs at `/opt/tokenizers/` (INFR-78). That's the whole delta — base image is otherwise untouched.

`Dockerfile.upstream`, `install.sh`, `build.sh`, and `config.py` are leftovers from a deprecated build approach (vendoring jetson-containers' install scripts). They are no longer driven by the build; INFR-91 documents why that path can't produce a working image standalone. Marked unused in `sglang/orin/README.md` Files table; safe to delete after one release cycle if nobody revisits.

## Commands

### CI build (preferred)

CI is `.github/workflows/build-sglang.yml`, runs on `ubuntu-24.04-arm` (Graviton SBSA, native aarch64 — no QEMU). Triggered on push to `main` under `sglang/**` or workflow-yml changes, on tags `v*`, on PRs, and via `workflow_dispatch`. Pushes to `ghcr.io/infernode-os/serving-sglang:{orin,thor}-<sha>` and `:{orin,thor}-latest`.

`workflow_dispatch` exposes `orin_base_image` as an override input — use when exploring an alternate dustynv tag (e.g. for INFR-92's upgrade work).

### Manual Orin build (any aarch64 host)

```sh
cd sglang/orin
docker build \
  --build-arg BASE_IMAGE=dustynv/sglang:r36.4.0 \
  -t serving-sglang:orin-local .
```

If your build host runs a dedicated experimental Docker daemon on a non-default socket, prepend `--host unix:///run/<daemon>.sock` — that's a per-environment concern, not part of the build itself.

### Manual Thor build

```sh
docker build -t serving-sglang:thor-local sglang/thor/
```

### Smoke tests

The on-hardware validator launches `sglang.launch_server` with TinyLlama, asserts `/health`, asserts the `KV Cache is allocated` startup line, and exercises `/v1/chat/completions`. **Run it after every published-image pull** — CI's build-time guards can only check metadata (no GPU on the GitHub runner):

```sh
HF_CACHE="${HF_CACHE:-/var/lib/huggingface}"
docker run --rm \
  --runtime nvidia --gpus all \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  serving-sglang:orin-local \
  /opt/sglang/validate-on-hardware.sh
```

Expected last line: `All on-hardware checks passed.`. Serving smoke takes ~60s cold, ~15s warm. Deploy + healthcheck sequence is in `runbooks/deploy.md`.

## Operational coexistence with Ollama

SGLang **coexists** with Ollama on the deploy host, it does not replace it. `lucibridge` routes per-request based on tool-category:

- `limbo_authoring` → Ollama (Devstral, single-user fluency)
- `dispatch` / `tool_call` / `memory` / `task` → SGLang (gpt-oss, concurrent fan-out, xgrammar)
- unknown / unset → Ollama (backward-compatible fallback)

The routing config schema lives at `runbooks/lucibridge-routing.md`; the bridge implementation lives in `infernode-os/infernode` (cross-repo). When changing the routing schema here, the consuming agentlib code in that repo needs a coordinated change — see INFR-79.

A pre-INFR-79 single-`LLM_BACKEND_URL` config must keep working unchanged. The bridge constructs an implicit single-backend config when `/etc/lucibridge/routing.json` is absent; do not break this fallback.

## Key files

| Path | What it is |
|---|---|
| `sglang/orin/Dockerfile` | Thin delta on `dustynv/sglang:r36.4.0` — dataclasses-backport removal, torch.compile-disable ENV, tokenizer bake. That's the whole delta |
| `sglang/orin/validate-on-hardware.sh` | On-hardware validator with real `/v1/chat/completions` smoke + KV-cache assertion — the safety net for any base-image bump |
| `sglang/orin/bake-tokenizers.sh` | InferNode-authored — bakes Llama-3 / Llama-3.1 tokenizer dirs into `/opt/tokenizers/` (INFR-78) |
| `sglang/orin/{Dockerfile.upstream,install.sh,build.sh,config.py}` | Leftovers from the deprecated framework-vendoring path; not driven by the build. Slated for removal after one release cycle |
| `sglang/LICENSE-UPSTREAM.md` | Records the dusty-nv commit SHA at vendoring; consult before re-syncing |
| `docs/SGLANG-ADOPTION-NOTES.md` | Spike findings, measured bake-off (SGLang ~78 tok/s @ N=8 vs Ollama's ~23 tok/s plateau), the canonical working recipe |
| `runbooks/deploy.md` | End-to-end deploy guide for a clean Orin AGX; acceptance gate is "new contributor brings SGLang up in <15 min" |
| `runbooks/lucibridge-routing.md` | Routing config schema + per-tool / per-category mapping |

## Conventions worth knowing

- **Launch arguments live in the runbook, not the image.** No `CMD` / `ENTRYPOINT` is baked into the production Dockerfiles so the same image can serve different model paths without rebuild. When changing launch flags, update `runbooks/deploy.md` §3 and §5 (systemd unit) together.
- **Tokenizer path is mandatory for Llama-3 family, not for the TinyLlama smoke.** SGLang 0.4's GGUF tokenizer doesn't register Llama-3 special tokens; `--tokenizer-path /opt/tokenizers/llama-3.1` (baked by `bake-tokenizers.sh`) is the fix. TinyLlama uses Llama-2's tokenizer and doesn't need the override. See INFR-78.
- **`TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1` and `--disable-cuda-graph` are required on Jetson** — `torch.compile` is broken at-import in this Triton 3.2 + Torch 2.5 combo (it pulls in `torch._inductor.runtime.hints` which calls `fields(AttrsDescriptor)` on a non-dataclass), and CUDA-graph capture is unreliable on Tegra. The two env vars are baked into the image; don't remove without re-running the validator.
- **The dataclasses backport must stay removed.** dustynv ships a Python-2-era `dataclasses.py` in `/usr/local/lib/python3.10/dist-packages/` that shadows the stdlib. The Dockerfile moves it to `_disabled_backports/` and asserts the stdlib is now what resolves. If a future base-image bump re-introduces it, the same pattern works.
- **CI guards are metadata-only; `validate-on-hardware.sh` is the real gate.** GitHub-hosted runners have no GPU, so the in-Dockerfile guards can only check torch+CUDA metadata, triton import, and the sglang version pin. Real correctness lives in the on-hardware validator's serving smoke. Always run the validator after pulling a new tag onto the deploy hardware.
- **NGC pulls are anonymous by default.** The `NGC_API_KEY` step in CI is forward-compatible scaffolding for any future gated tag — don't make it a hard requirement.
