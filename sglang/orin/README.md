# sglang/orin/ — Jetson Orin AGX (sm_87)

Container recipe for SGLang on Jetson Orin AGX. Vendored from
`dusty-nv/jetson-containers:packages/llm/sglang/` (see
`../LICENSE-UPSTREAM.md`) with version pin and Orin-specific notes
diverged from upstream.

## Target

| Property | Value |
|---|---|
| GPU | Ampere Tegra (`sm_87`) |
| JetPack | 6.x (R36.4 series) |
| CUDA | 12.6 |
| cuDNN | 9.3 |
| L4T base | `r36.4.0` |
| Python | 3.10 |
| PyTorch | 2.5–2.6 (`USE_DISTRIBUTED=1`, `TORCH_CUDA_ARCH_LIST=8.7`) |

## Pinned version

`SGLANG_VERSION=0.5.3` — see the docstring at the top of `config.py`
for the rationale and fallback ladder. The first 0.5.x line with
`srt/models/gpt_oss.py`. On-target smoke build on Hephaestus is the
gate (INFR-77).

If the pinned version fails to build, fall back in this order:
0.5.2 → 0.5.1 → 0.5.0, then 0.4.5+. Document the working pin back in
`config.py`'s `package = [ … ]` and update this README's "Pinned
version" line.

## Build (CI, GitHub-hosted)

CI builds the container on `ubuntu-24.04-arm` (Graviton SBSA, native
aarch64 — no QEMU). See `.github/workflows/build-sglang.yml`. The
build cross-compiles CUDA kernels for `sm_87` via
`TORCH_CUDA_ARCH_LIST=8.7` at `nvcc` invocation; the output image
runs on Jetson Orin AGX (Tegra).

## Build (manual on Hephaestus)

When the CI image is unavailable or you're iterating on the recipe:

```sh
cd ~/serving/sglang/orin
docker build \
  --build-arg BASE_IMAGE=dustynv/pytorch:2.6-r36.4.0-cu126-22.04 \
  --build-arg SGLANG_VERSION=0.5.3 \
  --build-arg SGLANG_VERSION_SPEC=0.5.3 \
  --build-arg IS_SBSA=0 \
  -t serving-sglang:orin-local .
```

(The exact `BASE_IMAGE` tag depends on what dustynv has published at
the time. Use `dustynv/pytorch` rather than `dustynv/sglang` to avoid
inheriting their stale 0.4.1 install; we re-install our pinned 0.5.x
fresh via `install.sh` / `build.sh`.)

## Files

| File | Origin | Purpose |
|---|---|---|
| `config.py` | **modified** | jetson-containers package config; pinned to our Orin-compatible version |
| `Dockerfile` | **modified** | standalone build with tokenizer bake (INFR-78) |
| `Dockerfile.upstream` | verbatim from upstream | reference copy for diff against upstream re-syncs |
| `build.sh` | verbatim from upstream | source-build fallback if `pip install` fails |
| `install.sh` | verbatim from upstream | `pip install sglang[all]~=$SGLANG_VERSION` first-try path |
| `bake-tokenizers.sh` | InferNode-authored | downloads Llama-3 / Llama-3.1 tokenizer dirs into `/opt/tokenizers/` (INFR-78) |
| `test.py` | verbatim from upstream | smoke test (`import sglang`, print version + CUDA device) |

## What this does NOT include

* **A `BASE_IMAGE`**. The Dockerfile expects one to be passed at build
  time (matches the upstream jetson-containers pattern, which chains
  base images via its framework). The CI workflow supplies a
  Jetson-rooted base; for manual builds see the command above.
* **Tokenizer pre-bake**. The Llama-3 tokenizer fix lives at
  `sglang/orin/tokenizers/` once INFR-78 lands. Until then,
  `--tokenizer-path` must be set at launch time.
* **Entrypoint scripts**. Launch arguments live with the runbook
  (`runbooks/hephaestus-deploy.md`) rather than baked into the image,
  so the same image serves different model paths without rebuild.

## Verifying a build

After a successful image build, run the upstream smoke test inside the
container:

```sh
docker run --rm --gpus all --runtime nvidia serving-sglang:orin-local \
  python3 /opt/sglang/test.py
```

Expected output:

```
testing SGLang...
✅ Memory cleared
SGLang version: 0.5.3
CUDA available: True
CUDA device: Orin (or NVIDIA Jetson AGX Orin)
SGLang OK
```

`gpt-oss` arch verification (per INFR-77 acceptance):

```sh
docker run --rm --gpus all --runtime nvidia serving-sglang:orin-local \
  python3 -c "import sglang.srt.models.gpt_oss as m; print('gpt_oss arch module:', m.__file__)"
```
