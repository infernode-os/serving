# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Container builds, deployment recipes, and routing configuration for the LLM serving stack behind InferNode (`llmsrv` / `lucibridge` / `serve-llm.sh`). It does **not** contain the OS runtime (that lives in `infernode-os/infernode`) or the training pipeline (that lives in `pdfinn/infernode-os-llm`). Lifecycle here tracks JetPack releases / SGLang versions / hardware generations, not model training cycles.

The repo is intentionally **public** (changed 2026-05-14) so CI can use the free `ubuntu-24.04-arm` GitHub-hosted runner minutes. Do not commit secrets or credential-shaped files.

Work is tracked under the **INFR Jira project's "Productize SGLang serving" epic** (parent: INFR-68). When a file references `INFR-NN`, that's a Jira ticket — see ticket commentary for context Claude can't recover from the tree.

## Hardware-target asymmetry (load-bearing)

The `sglang/` subtree has two variants because the two Jetson generations need different base-image strategies:

- **`sglang/orin/`** — Jetson Orin AGX (`sm_87`, JetPack 6.x, CUDA 12.6). NGC has no SGLang image for this combination, so we **vendor** the `dusty-nv/jetson-containers` SGLang recipe (MIT) and pin our own SGLang version. The Orin Dockerfile expects a `BASE_IMAGE` build-arg pointing at a `dustynv/pytorch` tag (NOT `dustynv/sglang` — we re-install fresh to avoid inheriting their stale SGLang).
- **`sglang/thor/`** — Jetson Thor (`sm_103`, JetPack 7.x, CUDA 13+). NVIDIA's official `nvcr.io/nvidia/sglang:25.10-py3` targets this hardware, so the Dockerfile is a thin wrapper that adds only InferNode-specific bits. Forward-looking — no Thor box exists yet.

The Orin recipe vendors upstream verbatim *except* `config.py` (pinned version) and `Dockerfile` (standalone build + tokenizer bake). `Dockerfile.upstream` is kept as a verbatim copy specifically so re-syncs from upstream are a clean diff. When re-syncing, **never auto-overwrite `config.py`** — its divergence is intentional.

## Commands

### CI build (preferred)

CI is `.github/workflows/build-sglang.yml`, runs on `ubuntu-24.04-arm` (Graviton SBSA, native aarch64 — no QEMU). Triggered on push to `main` under `sglang/**` or workflow-yml changes, on tags `v*`, on PRs, and via `workflow_dispatch`. Pushes to `ghcr.io/infernode-os/serving-sglang:{orin,thor}-<sha>` and `:{orin,thor}-latest`.

`workflow_dispatch` inputs let you override `orin_base_image` and `sglang_version` without code changes — use this when bumping SGLang or chasing a new dustynv base tag.

### Manual Orin build (on Hephaestus or any aarch64 host with the right base)

```sh
cd sglang/orin
docker build \
  --build-arg BASE_IMAGE=dustynv/pytorch:2.6-r36.4.0-cu128-24.04 \
  --build-arg SGLANG_VERSION=0.5.3 \
  --build-arg SGLANG_VERSION_SPEC=0.5.3 \
  --build-arg IS_SBSA=0 \
  -t serving-sglang:orin-local .
```

If the pinned 0.5.3 fails to build, fall back: 0.5.2 → 0.5.1 → 0.5.0 → 0.4.5+. Document the working pin in `sglang/orin/config.py`'s `package = [ … ]` and update `sglang/orin/README.md`'s "Pinned version" line.

### Manual Thor build

```sh
docker build -t serving-sglang:thor-local sglang/thor/
```

### Smoke tests (run inside the built container)

```sh
# Basic SGLang import + CUDA visibility
docker run --rm --gpus all --runtime nvidia serving-sglang:orin-local \
  python3 /opt/sglang/test.py

# gpt-oss model class present (INFR-77 acceptance gate)
docker run --rm --gpus all --runtime nvidia serving-sglang:orin-local \
  python3 -c "import sglang.srt.models.gpt_oss as m; print(m.__file__)"
```

End-to-end smoke testing happens **manually on Hephaestus** after each successful CI build — GitHub's hosted runners have no Jetson hardware. The deploy + healthcheck sequence is in `runbooks/hephaestus-deploy.md`.

## Hephaestus dual-purpose model (do NOT violate — INFR-90)

Hephaestus serves two purposes that pull in opposite directions:

1. **Field-parity reference** — TAK / NERVA / Ollama must run under a Docker daemon whose storage lives on the device's only disk (root partition), exactly as they would on a no-SSD field unit.
2. **Development environment** — experimental work (SGLang, future Jetson experiments) needs the 916 GiB `/mnt/orin-ssd` and would crush the root partition if forced through the production daemon.

A single Docker daemon has one storage root, so the only honest answer is **two daemons**:

| Daemon | Socket | Storage | Used by |
|---|---|---|---|
| `docker.service` (production) | `/run/docker.sock` | root partition (with the Docker 29 quirk that its image content lives at `/var/lib/containerd` on root, regardless of where `/var/lib/docker` symlinks to) | TAK · NERVA · Ollama · field-parity workloads |
| `docker-dev.service` (experimental) | `/run/docker-dev.sock` | `/mnt/orin-ssd/docker-dev` (own legacy snapshotter, `containerd-snapshotter: false`) | SGLang · any other Jetson experiment |

**Rules for Claude when working on this host:**

- For SGLang or any experimental work, use the dev daemon: `docker --host unix:///run/docker-dev.sock` (or set up `alias dev-docker=...`).
- **Never** pull SGLang or other experimental images on the production daemon — its 12-14 GiB image plus working set would crush root.
- **Never** migrate TAK / NERVA / Ollama to the dev daemon — they belong on production for field parity.
- The runbook §0.1 has the full one-time setup if the dev daemon ever needs to be re-created.
- Pre-flight any plan that involves `docker pull`/`docker run` by checking *which* daemon you're hitting. The default `docker` CLI uses the production daemon — easy to forget on Hephaestus.

### Watch out: `/var/lib/docker` symlink is misleading

`/var/lib/docker` is symlinked to `/mnt/orin-ssd/docker/docker` on this host, but `/var/lib/containerd` (the actual image content store under Docker 29's containerd integration) lives on root. So the symlink only redirects the daemon's *metadata*, not its *image content*. A pull on the production daemon will land on root regardless of what the symlink suggests. This was discovered the hard way (root went from 12 GiB → 877 MB free during a base-image pull); the dev daemon avoids the trap by disabling the containerd snapshotter entirely.

## Operational coexistence with Ollama

SGLang **coexists** with Ollama on Hephaestus, it does not replace it. `lucibridge` routes per-request based on tool-category:

- `limbo_authoring` → Ollama (Devstral, single-user fluency)
- `dispatch` / `tool_call` / `memory` / `task` → SGLang (gpt-oss, concurrent fan-out, xgrammar)
- unknown / unset → Ollama (backward-compatible fallback)

The routing config schema lives at `runbooks/lucibridge-routing.md`; the bridge implementation lives in `infernode-os/infernode` (cross-repo). When changing the routing schema here, the consuming agentlib code in that repo needs a coordinated change — see INFR-79.

A pre-INFR-79 single-`LLM_BACKEND_URL` config must keep working unchanged. The bridge constructs an implicit single-backend config when `/etc/lucibridge/routing.json` is absent; do not break this fallback.

## Key files

| Path | What it is |
|---|---|
| `sglang/orin/config.py` | jetson-containers package config; **intentionally divergent** from upstream — pinned to our Orin-compatible SGLang version |
| `sglang/orin/Dockerfile` | Standalone production build (modified from upstream) |
| `sglang/orin/Dockerfile.upstream` | Verbatim upstream copy, kept for diff against re-syncs |
| `sglang/orin/bake-tokenizers.sh` | InferNode-authored — bakes Llama-3 / Llama-3.1 tokenizer dirs into `/opt/tokenizers/` (INFR-78 fix for GGUF tokenizer not registering Llama-3 special tokens) |
| `sglang/LICENSE-UPSTREAM.md` | Records the dusty-nv commit SHA at vendoring; consult before re-syncing |
| `docs/SGLANG-ADOPTION-NOTES.md` | Spike findings, measured bake-off (SGLang ~78 tok/s @ N=8 vs Ollama's ~23 tok/s plateau), original launch flags |
| `runbooks/hephaestus-deploy.md` | End-to-end Hephaestus deploy; acceptance gate is "new contributor brings SGLang up in <15 min" |
| `runbooks/lucibridge-routing.md` | Routing config schema + per-tool / per-category mapping |

## Vendored install-script gotchas (Orin Dockerfile)

The dusty-nv `install.sh` / `build.sh` we vendor were designed to run *inside* the upstream jetson-containers framework, which provides a long chain of prereqs implicitly. Three of them bit us when we tried to build standalone, and the fixes all live in our `Dockerfile`:

1. **`uv` is not on PATH in the base image.** Upstream chains a `uv`-install package as a previous stage. We `pip install uv` ourselves before invoking `install.sh`.
2. **`PIP_WHEEL_DIR` is not exported.** Upstream sets it from a parent stage. We export it via `ENV` in our Dockerfile.
3. **`uv` ignores `PIP_INDEX_URL`** — it only reads `UV_INDEX_URL` / `UV_EXTRA_INDEX_URL`. The dustynv base image sets `PIP_INDEX_URL` to the JetPack mirror (`https://pypi.jetson-ai-lab.dev/jp6/cu128`) which has Jetson-built CUDA wheels for torch/triton with `sm_87` device code. Without `UV_INDEX_URL` set, `uv pip install sglang[all]` resolves torch from upstream PyPI and pulls the **CPU wheel**, silently clobbering the base image's CUDA torch. The image then runs but reports `cuda False` on a Tegra GPU. We set `UV_INDEX_URL` + `UV_EXTRA_INDEX_URL` + `UV_INDEX_STRATEGY=unsafe-best-match` at the Dockerfile layer.

**General rule when vendoring jetson-containers recipes:** verify every implicit framework dependency. Things to grep for in the vendored scripts that signal an implicit prereq: `uv` (the installer), `PIP_WHEEL_DIR`, `TORCH_CUDA_ARCH_LIST`, `IS_SBSA`, `FORCE_BUILD`, anything ending in `_VERSION`. Set them at the Dockerfile layer rather than patching the vendored scripts (keeps the diff against upstream clean).

## Conventions worth knowing

- **Launch arguments live in the runbook, not the image.** No `CMD` / `ENTRYPOINT` is baked into the production Dockerfiles so the same image can serve different model paths without rebuild. When changing launch flags, update `runbooks/hephaestus-deploy.md` §3 and §5 (systemd unit) together.
- **Tokenizer path is mandatory for Llama-3 family.** SGLang 0.4/0.5's GGUF tokenizer doesn't register Llama-3 special tokens; `--tokenizer-path /opt/tokenizers/llama-3.1` (baked by `bake-tokenizers.sh`) is the fix. See INFR-78.
- **`TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1` and `--disable-cuda-graph` are required on Jetson** — `torch.compile` and CUDA-graph capture are unreliable on Tegra. Don't remove these without re-running the spike.
- **The `IS_SBSA=0` build-arg matters.** Jetson Tegra (`IS_SBSA=0`) and datacenter ARM SBSA (`IS_SBSA=1`) take different code paths in `install.sh` / `build.sh`. Orin builds always pass `IS_SBSA=0`.
- **NGC pulls are anonymous by default.** The `NGC_API_KEY` step in CI is forward-compatible scaffolding for any future gated tag — don't make it a hard requirement.
