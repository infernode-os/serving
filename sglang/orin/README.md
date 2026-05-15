# sglang/orin/ — Jetson Orin AGX (sm_87)

Container recipe for SGLang on Jetson Orin AGX. Built as a **delta upgrade** on top of dustynv's pre-built SGLang base image — see `Dockerfile` for the full rationale, or `INFR-91` for the investigation that ruled out the previous vendored-install path.

## Target

| Property | Value |
|---|---|
| GPU | Ampere Tegra (`sm_87`) |
| JetPack | 6.x (R36.4 series) |
| CUDA (container) | 12.9 |
| L4T base | `r36.4` |
| Python (container) | 3.12 |
| PyTorch | 2.8.0+cu129 (sm_87, from dustynv base) |
| Triton | 3.4.0 |
| SGLang | **0.4.9** (dustynv-bundled; v1 baseline — see "Pinned versions" below) |
| sgl-kernel | 0.2.3 (dustynv-built for sm_87) |

## Pinned versions — v1 ships with dustynv's bundled SGLang 0.4.9

This is a deliberate hold-back from SGLang 0.5.x. The constraint chain (full investigation in [INFR-91] + [INFR-92]):

* SGLang 0.5.x strictly pins `sgl-kernel==0.3.16.post3`.
* `sgl-kernel 0.3.16.post3`'s PyPI aarch64 wheel ships **only sm_90 (Hopper) and sm_100 (Blackwell) binaries** — no sm_87 (Orin). Cross-arch `.so` loading succeeds at import time, but the first kernel call hits `RuntimeError: no kernel image is available for execution on the device`.
* Building sgl-kernel 0.3.16.post3 from source for sm_87 is blocked by an upstream broken deepgemm commit SHA in sgl-kernel's CMake `FetchContent` declaration.
* Workarounds (mixing dustynv 0.2.3 sm_87 binaries with sglang 0.5.x, `SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1`) all hit either API gaps or dispatch machinery that ignores the override.

So v1 ships dustynv's bundled SGLang 0.4.9 which uses sgl-kernel 0.2.3 (built for sm_87 by dustynv). The server actually starts and serves requests. Trade-off: **no `gpt_oss.py` model arch in 0.4.x**, so lucibridge's gpt-oss routing falls back to Ollama on Orin. The fallback was designed for exactly this case (per `runbooks/lucibridge-routing.md`); production is degraded but not broken.

Tracking the upgrade to 0.5.x + gpt_oss support: [INFR-92].

[INFR-91]: https://nervsystems-team.atlassian.net/browse/INFR-91
[INFR-92]: https://nervsystems-team.atlassian.net/browse/INFR-92

## Base image rationale

The base is `dustynv/sglang:r36.4-cu129-24.04` (not `dustynv/pytorch`). Earlier (pre-INFR-91) we used `dustynv/pytorch` and tried to install SGLang on top via the vendored `dusty-nv/jetson-containers` recipe. That path could not produce a working image because:

* SGLang 0.5.x strictly pins `torch==2.8.0`.
* `torch==2.8.0 + cu128/9 + sm_87 + cp312` has no public wheel on upstream PyPI (cp310 only), NVIDIA's developer redist (tops out at 2.4), or `pypi.jetson-ai-lab.io`'s JP6 mirror (cp310 only).
* uv resolved torch from upstream PyPI's CPU wheel, silently clobbering the dustynv base's CUDA torch — the resulting image imported SGLang but reported `cuda False` on a Tegra GPU.

The only build of `torch 2.8 + cu129 + sm_87 + cp312` in existence is *bundled inside* dustynv's `r36.4-cu129-24.04` SGLang image. So we use that image as base and delta-upgrade dustynv's bundled SGLang 0.4.9 → 0.5.4. torch is left alone.

## Build (CI, GitHub-hosted)

CI builds on `ubuntu-24.04-arm` (Graviton SBSA, native aarch64 — no QEMU). See `.github/workflows/build-sglang.yml`. Six build-time guards at the end of the Dockerfile fail the build if torch loses CUDA, or if any of `triton` / `sgl_kernel` / `sglang` / `gpt_oss` / `launch_server` fail to import — would have caught all the bugs we hit during INFR-91.

## Build (manual on Hephaestus)

```sh
cd ~/serving/sglang/orin
docker --host unix:///run/docker-dev.sock build \
  --build-arg BASE_IMAGE=dustynv/sglang:r36.4-cu129-24.04 \
  -t serving-sglang:orin-local .
```

(Hephaestus-specific: use the dev daemon socket via `--host` per the dual-daemon policy in `runbooks/hephaestus-deploy.md` §0.1. On a field-deployment Orin AGX, drop the `--host` flag.)

## Files

| File | Origin | Purpose |
|---|---|---|
| `Dockerfile` | **InferNode-authored** | Delta-upgrade recipe over dustynv/sglang |
| `bake-tokenizers.sh` | InferNode-authored | Downloads Llama-3 / Llama-3.1 tokenizer dirs into `/opt/tokenizers/` (INFR-78) |
| `test.py` | verbatim from `dusty-nv/jetson-containers` | Smoke test (`import sglang`, print version + CUDA device) |
| `Dockerfile.upstream` | verbatim from `dusty-nv/jetson-containers` | **Unused.** Reference copy from when we tried the framework's install.sh path. Kept for diff against upstream re-syncs if we ever revisit. |
| `config.py` | from `dusty-nv/jetson-containers`, **was modified** | **Unused.** jetson-containers package config; no longer driven by our build. |
| `install.sh` / `build.sh` | verbatim from `dusty-nv/jetson-containers` | **Unused.** The vendored install scripts that drove the previous build path. INFR-91 documents why they couldn't produce a working image standalone. |

The "unused" files are retained for now under MIT attribution (`../LICENSE-UPSTREAM.md`). They can be deleted in a future cleanup if no one revisits the framework path within ~1 release cycle.

## What this does NOT include

* **Entrypoint scripts.** Launch arguments live with the runbook (`runbooks/hephaestus-deploy.md`) rather than baked into the image, so the same image serves different model paths without rebuild.

## Verifying a build

```sh
docker --host unix:///run/docker-dev.sock run --rm --gpus all --runtime nvidia \
  ghcr.io/infernode-os/serving-sglang:orin-latest \
  python3 /opt/sglang/test.py
```

Expected:

```
testing SGLang...
✅ Memory cleared
SGLang version: 0.4.9
CUDA available: True
CUDA device: Orin
SGLang OK
```

`gpt-oss` arch verification (deferred to INFR-92 — v1 ships SGLang 0.4.9 without gpt_oss.py):

```sh
# Currently EXPECTS to fail — gpt_oss.py is in SGLang 0.5+ and v1 ships 0.4.9.
# Tracked: INFR-92.
docker --host unix:///run/docker-dev.sock run --rm --gpus all --runtime nvidia \
  ghcr.io/infernode-os/serving-sglang:orin-latest \
  python3 -c "import sglang.srt.models.gpt_oss" || echo "(expected for v1; INFR-92)"
```

End-to-end launch:

```sh
docker --host unix:///run/docker-dev.sock run --rm --gpus all --runtime nvidia \
  ghcr.io/infernode-os/serving-sglang:orin-latest \
  python3 -m sglang.launch_server --help
```

Should print the launch_server CLI with `--model-path`, `--tokenizer-path`, etc.
