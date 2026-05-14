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
| SGLang | 0.5.4 (delta upgrade from base's 0.4.9) |
| sgl-kernel | 0.3.16.post3 |

## Pinned versions

`SGLANG_VERSION=0.5.4` + `SGL_KERNEL_VERSION=0.3.16.post3`. The pin is constrained by torch availability: every SGLang 0.5.x strictly pins `torch==2.8.0`, and the only place torch 2.8 + cu128/9 + sm_87 + cp312 exists as a working wheel is **inside dustynv's `r36.4-cu129-24.04` image**. The 0.5.4 sgl-kernel post-3 is the version that ships with sglang 0.5.4. INFR-77 acceptance gate (`gpt_oss.py` present) is met at 0.5.x.

If a newer SGLang is desired, the constraint chain to verify is: (1) a dustynv tag exists with the required torch for that SGLang; (2) the sgl-kernel pin for that SGLang has a wheel for the base's torch + cu + cp.

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
  --build-arg SGLANG_VERSION=0.5.4 \
  --build-arg SGL_KERNEL_VERSION=0.3.16.post3 \
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
SGLang version: 0.5.4
CUDA available: True
CUDA device: Orin
SGLang OK
```

`gpt-oss` arch verification (INFR-77 acceptance):

```sh
docker --host unix:///run/docker-dev.sock run --rm --gpus all --runtime nvidia \
  ghcr.io/infernode-os/serving-sglang:orin-latest \
  python3 -c "import sglang.srt.models.gpt_oss as m; print('gpt_oss arch module:', m.__file__)"
```

End-to-end launch:

```sh
docker --host unix:///run/docker-dev.sock run --rm --gpus all --runtime nvidia \
  ghcr.io/infernode-os/serving-sglang:orin-latest \
  python3 -m sglang.launch_server --help
```

Should print the launch_server CLI with `--model-path`, `--tokenizer-path`, etc.
