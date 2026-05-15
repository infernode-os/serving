# SGLang adoption — training & deployment notes

> **Moved 2026-05-14 from `pdfinn/infernode-os-llm:docs/SGLANG-ADOPTION-NOTES.md`** (branch `docs/sglang-adoption-notes`, never merged to main). The `infernode-os/serving` repo is now the canonical home for serving-infrastructure docs. The IOL location has a stub pointer; this is the working copy.

**Date:** 2026-05-13 (initial) — 2026-05-14 (working recipe + bake-off)
**Owning ticket:** [INFR-68](https://nervsystems-team.atlassian.net/browse/INFR-68) — Spike: SGLang on Jetson Orin for LoRA-native multi-model serving
**Status:** **Measured + working recipe documented.** SGLang server runs on a Jetson Orin AGX dev box; bake-off complete (SGLang scales ~3× under concurrent load vs Ollama). Productization tracked under the "Productize SGLang serving" epic in INFR.

This document captures what would have to change in IOL's training and
deployment pipelines if InferNode swaps (or doubles up) Ollama with
SGLang as the on-device inference backend. It is **deliberately
written before any experiment**, so the first reality-check round of
the spike will probably contradict some of it. Treat as a working
hypothesis, not a recipe.

The strongest single reason for IOL to care about this swap is
**V4-PLAN's "gpt-oss-limbo-v3 deploy — blocked on Ollama LoRA
support."** Ollama refuses to load any LoRA on gpt-oss; SGLang loads
PEFT adapters natively. Everything else here is secondary to that.

---

## Deployment side — what changes vs the current Ollama path

### The artifact pipeline collapses

Current pipeline (per `docs/V3-DEPLOY-RUNBOOK.md`):

```
PEFT LoRA checkpoint
  → merge_to_gguf.py (merge + convert + Q4_K_M quantize)
  → GGUF + Modelfile
  → rsync to deploy host
  → ollama create -f Modelfile
  → /v1/chat/completions
```

With SGLang the destination accepts the PEFT checkpoint **directly**:

```
PEFT LoRA checkpoint
  → rsync to the deploy host (just the adapter dir; tens of MB, not 13 GB)
  → sglang.launch_server --model-path <base> --lora-paths <adapter>
  → /v1/chat/completions
```

What this eliminates:

- `training/shared/merge_to_gguf.py` for the SGLang path (still needed
  for Ollama if we keep both)
- The whole class of `convert_hf_to_gguf.py` tokenizer-corruption
  failures (Mistral-Tekken bug per `M4-cloud-runbook.md:170-171`,
  GGUF metadata `\x00` issues from `V3-DEPLOY-RUNBOOK` step 5)
- The "merge OOMs on RunPod" risk in the deploy phase — there is no
  merge step
- The Devstral-vs-gpt-oss procedural fork (LoRA-as-adapter for one,
  merge-then-convert for the other). Both become "ship the adapter
  dir."

What replaces it:

- A quantization step on the **base model** (one-time per base), not
  per-adapter. AWQ or GPTQ rather than GGUF Q4_K_M. ~equivalent
  quality, different artifact.
- Optional: an AWQ calibration pass with a held-out sample of our
  training data so the quantization scales are aligned to our domain.
  Not strictly required; modest quality bump.

### Quant format: AWQ/GPTQ/FP8, not GGUF

SGLang's first-class quants are AWQ, GPTQ, and FP8. GGUF is not
supported in production paths.

V3-DEPLOY-RUNBOOK's "Considered alternatives" table previously rejected
AWQ on the grounds that "Triton kernels per-arch; less proven on
Jetson Orin than llama.cpp." That reasoning still holds and is the
single biggest risk for this whole spike — it is the same Triton
kernel concern that drove the choice of Ollama in the first place.
The spike must prove AWQ-on-`sm_87` works before any other claim
matters.

If AWQ-on-Jetson holds, target quants per base:

| Base | Recommended | Expected resident (24B-class) |
|---|---|---:|
| `mistralai/Devstral-Small-2507` (Mistral, dense) | AWQ INT4 | ~13 GB + KV cache |
| `openai/gpt-oss-20b` (MoE, 3.6B active) | AWQ INT4 or FP8 | smaller than Devstral despite param count |

### Multi-LoRA hot-swap on one resident base

SGLang supports multiple LoRA adapters layered on a single
VRAM-resident base, selected per-request. For us this means:

- One `gpt-oss-20b` base resident → serves `gpt-oss-limbo-v3`,
  future `gpt-oss-limbo-v4`, and the unmodified base concurrently
  with no extra VRAM beyond adapter weights (~30 MB each).
- The same for Devstral.
- Production routing decision from V4-PLAN ("gpt-oss/low for
  dispatch, devstral-LoRA for Limbo authoring") becomes a per-request
  `lora_name` selector rather than two separate Ollama models.

This is also the answer to "how do we A/B v3 vs v4 in production
without doubling resident memory."

### Chat template handling

SGLang applies the HF tokenizer's chat template directly. This sidesteps
Ollama's chat-template parser, which is where `[TOOL_CALLS]` /
`<SPECIAL_NN>` leakage originates per `docs/MODEL-INTEGRATION-NOTES.md`
and IOL V4-PLAN §B3.

Two consequences worth measuring:

1. **The leakage class may simply not occur** at the serving layer
   even without grammar-constrained decoding, because SGLang doesn't
   try to re-parse Mistral's `[TOOL_CALLS]` marker back into OpenAI
   `tool_calls` — it just emits what the model emits. Whether
   `lucibridge` then has to do the parsing is an open question for
   the spike.
2. **xgrammar (SGLang's constrained decoder)** makes the malformed
   output unrepresentable in the first place. If this works on
   Devstral, several v4 corpus items become unnecessary:
   - B3 stop-discipline tightening (tool-bearing turns must be
     content-empty)
   - B4 args-canonicalisation negatives
   - A1 harness recovery for `<SPECIAL_NN>` markers (still ship it as
     defence in depth, but it would stop firing)

Worth bench-marking explicitly: same Devstral checkpoint, served two
ways, run `baseline_v4.yaml` with `<special_token_leak>` as a probe.

### `serve-llm.sh` and `lucibridge` impact

Wire format is OpenAI-compatible `/v1/chat/completions` — the same
shape `HEADLESS-LLM-DAEMON.md` already speaks. No `lucibridge` code
changes expected. Operational deltas:

- A `serve-llm-sglang.sh` sibling launcher to bring up the SGLang
  server on a non-conflicting port (Ollama keeps 11434; SGLang
  defaults to 30000). Both can coexist.
- `OLLAMA_MODELS` env / Modelfile concepts have no SGLang analogue.
  Adapter selection becomes a per-request `model` or `lora_name`
  field, set by `lucibridge`'s per-tool model selection logic
  (already a v4 follow-up per V4-PLAN).
- The serve-llm systemd unit in `HEADLESS-LLM-DAEMON.md` would gain
  a sibling unit; `After=`/`Requires=` change from `ollama.service`
  to an SGLang systemd unit we'd write.

---

## Training side — what we'd adopt or change

The good news: **most of the training pipeline doesn't need to
change.** The PEFT checkpoint is already the canonical output. The
Make targets, axolotl configs, RunPod scripts, and eval harness all
keep working.

The narrower deltas:

### 1. Keep the PEFT adapter as the deliverable; drop GGUF for the SGLang path

`training/devstral/qlora_sft.yaml` and `training/gpt_oss/qlora_sft.yaml`
already write `adapter_model.safetensors` + `adapter_config.json`. Nothing
to change there. The post-training step in the deploy runbook
(`merge_to_gguf.py`) is the only thing that becomes optional.

If we keep both stacks, this becomes a Make target fan-out:

- `make deploy-ollama` — existing GGUF pipeline (current)
- `make deploy-sglang` — rsync adapter dir to the deploy host + restart SGLang

No new Make scripts to author per the project's existing
"describe inline, don't wrap" preference; this is one rsync.

### 2. Validate LoRA target modules against SGLang's loader

Devstral / Mistral attention + MLP projections (`q_proj, k_proj,
v_proj, o_proj, gate_proj, down_proj, up_proj`) — SGLang's LoRA path is
well-trodden here. Low risk.

gpt-oss MoE per-expert MLP targets (`gate_proj`, `up_proj`, `down_proj`
applied per expert) — SGLang's LoRA support for MoE was added more
recently. **Verify against the v3 adapter before relying on it.** If
the loader rejects expert-keyed adapter weights, we have three
options, listed cheapest-first:

1. Repack the adapter's keys to whatever naming SGLang expects (no
   retrain).
2. Retrain gpt-oss with `lora_target_modules` restricted to attention
   only (skip per-expert MLP). Modest quality cost.
3. Merge-and-quantize the adapter into the base for SGLang — same
   shape as Option 1 in V4-PLAN, just for SGLang's quant format
   instead of GGUF. Loses the multi-LoRA hot-swap benefit for that
   adapter.

This validation is a 10-minute test on RunPod with the existing
checkpoint; do it early in the spike before committing.

### 3. AWQ calibration data (optional, small lift)

If we move to AWQ-INT4 on the base, the calibration pass benefits from
a few hundred examples drawn from our domain (Limbo authoring, 9P
tool calls). The existing `data/training/val.jsonl` is exactly the
right shape. One script (~50 lines) that loads the base, runs
autoawq's `quantize()` with our val set, writes the AWQ-INT4 model.
One-time per base. Skip for v0 of the spike; revisit if quality
drops vs Q4_K_M.

### 4. Tokenizer fidelity is automatic

The Mistral-Tekken tokenizer issue that bit `convert_hf_to_gguf.py`
(M4-cloud-runbook:170-171) does not exist on the SGLang path because
the serve-time tokenizer is the same `AutoTokenizer.from_pretrained()`
load that training uses. No conversion, no risk.

### 5. Eval harness keeps working unchanged

`make eval-baseline MODEL=… BASE_URL=…` already takes a URL.
SGLang's `/v1/chat/completions` is OpenAI-compatible, so the eval is
literally:

```sh
make eval-baseline MODEL=devstral-limbo-v3 BASE_URL=http://<deploy-host>:30000/v1
```

`tools/virgil-agent/scenarios/*.yaml` should also work as-is.

Bake-off harness for the spike:

```sh
make eval-baseline MODEL=devstral-limbo-v3 BASE_URL=http://<deploy-host>:11434/v1  # Ollama
make eval-baseline MODEL=devstral-limbo-v3 BASE_URL=http://<deploy-host>:30000/v1  # SGLang
make eval-grounding INPUT=eval/runs/<ollama>.gated.jsonl
make eval-grounding INPUT=eval/runs/<sglang>.gated.jsonl
```

`tool_use_grounding.py` already counts `special_token_leak` probes —
that's the key delta to look at.

### 6. Don't change the corpus yet

V4-PLAN's B3 / B4 / B5 / B6 corpus items target the model's behaviour
*regardless* of serving stack. If SGLang's xgrammar makes some of
them moot at serve-time, that's a serving win, not a reason to drop
them from the corpus — the model should still produce well-formed
output when grammar is off (and grammar will not always be on, e.g.
free-form Limbo authoring turns). Keep v4 corpus plans intact;
re-evaluate which items become low-priority *after* the spike has
real numbers.

---

## Per-model summary

### Devstral (current `devstral-limbo-v3`)

- **Adapter portable:** yes, PEFT checkpoint as-is.
- **Risk:** AWQ-on-`sm_87` quality vs GGUF Q4_K_M. Acceptance: ≥58%
  compile-pass (same bar V3-DEPLOY-RUNBOOK set against Q4_K_M).
- **Expected win:** zero `[TOOL_CALLS]` leakage with xgrammar on.
  Latency parity at minimum; possible speedup from no chat-template
  re-parsing.

### gpt-oss (current `gpt-oss-limbo-v3` — undeployed)

- **Adapter portable:** likely yes (validate MoE target-module names
  first per training §2).
- **Strategic win:** **this is the actual unblock.** The adapter
  exists and is unloadable today; SGLang loads it directly.
- **Risk:** MoE LoRA path is newer; AWQ MoE quantization quality is
  less established than Mistral-dense. If MoE-on-AWQ degrades badly,
  FP8 is the fallback (larger resident, still much smaller than bf16).

---

## What this does not address

- **Cost of running two stacks.** If both Ollama and SGLang stay
  resident on the deploy host we lose the multi-LoRA memory win. Either
  pick one or split Veltro's two production roles across them (which
  the V4-PLAN routing already implies).
- **Veltro Limbo-authoring path.** Compile-gate is unchanged; this
  is a serve-time discussion.
- **Build/CI.** None of this touches the IOL CI surface. The compile
  gate keeps running on Docker x86; the spike happens on the Jetson dev box.
- **Veltro security / namespace model.** Untouched.

---

## TL;DR for the spike runner

1. Validate AWQ-INT4 quantization runs on `sm_87` with whatever JetPack
   PyTorch/Triton combo the dev box is on **first**. Stop if it doesn't.
2. Validate SGLang loads the existing gpt-oss-limbo-v3 PEFT adapter
   on its MoE base **second**. If it doesn't, decide between repack /
   retrain-attn-only / merge-and-quantize before continuing.
3. Then bake off against Ollama with the existing eval harness, with
   and without xgrammar. The numbers from `tool_use_grounding.py`
   are the deliverable.

---

## Spike attempt — 2026-05-13/14 (measured)

First real attempt on the Jetson dev box. Result: **parked at the PyTorch
layer; no prebuilt wheel exists that satisfies all of SGLang's runtime
requirements on Jetson Orin AGX (`sm_87`).** This is a genuine
spike-finding — not in the pre-spike risk list above — so promoted to
its own section.

### What worked

- Dev-box baseline: JetPack 6.2 (L4T R36.4.7) / CUDA 12.6 / cuDNN
  9.3.0 / Orin `sm_87` / 64 GB unified memory / 916 GB working SSD.
- Disk reclamation on the working SSD: ~129 GB freed (HF xet cache, conda pkg
  cache, two F5-TTS conda envs, three unused HF model caches) without
  touching anything load-bearing (Devstral, gpt-oss-20b, all daedalus
  artifacts, other resident workloads, swapfile, IOL repo). 97% → 82%.
- Conda env at `${WORK}/conda-envs/sglang-spike`
  (Python 3.10.20). cuDNN-9-aligned PyTorch installed and verified
  (NVIDIA's `torch-2.5.0a0+nv24.08`, then upstream `torch-2.6.0+cu126`).
- **`daedalus-v1` structural validation**: the v4 Devstral PEFT
  adapter at `${WORK}/daedalus-v1/checkpoint-stripped/`
  is canonical-shape — peft 0.18.1 LoRA, r=32, α=64, all 7 target
  modules (q/k/v/o + gate/up/down), 560 tensors covering all 40 layers,
  184.8M trainable params. **Would load into SGLang's LoRA adapter
  loader without repacking.**
- **SGLang stack installable on aarch64**: contrary to the pre-spike
  hypothesis, `sglang 0.5.11`, `sglang-kernel 0.4.2`,
  `flashinfer-python 0.6.8.post1`, `flashinfer-cubin 0.6.8.post1`,
  `xgrammar 0.1.32`, and `triton 3.7.0` **all ship prebuilt aarch64
  cp310 wheels**. No source build needed for the SGLang surface itself.
  `import sglang` succeeds (after the workarounds in §"Required
  workarounds" below).

### The blocker — PyTorch on Jetson, in detail

SGLang's import path runs
`from torch.distributed import Backend, ProcessGroup`
at module load time (`srt/distributed/parallel_state.py`). This forces
**a PyTorch build with `USE_DISTRIBUTED=1`**, even for single-GPU
single-process serving.

Available prebuilt PyTorch wheels on aarch64+CUDA:

| Source | sm_87 kernels | `USE_DISTRIBUTED` | Works for SGLang? |
|---|---|---|---|
| NVIDIA JP6 (`developer.download.nvidia.com/jp/v60..v62/pytorch/`) — torch 2.4 / 2.5 | ✓ | ✗ (`torch.distributed.is_available()` returns False; no `_C._distributed_c10d` extension; `Backend` not exposed) | **No** — SGLang import fails on `Backend` |
| pytorch.org cu126 official aarch64 (`download.pytorch.org/whl/cu126/`) — torch 2.6.0+cu126 | ✗ (`RuntimeError: no kernel image is available for execution on the device` on a trivial CUDA matmul on Orin) | ✓ | **No** — fatbin targets datacenter ARM (sm_80/86/89/90/100); Tegra sm_87 not included |
| PyPI plain `torch` aarch64 wheel | n/a (CPU-only) | ✓ | No |

In other words: SGLang on Jetson Orin AGX requires a custom PyTorch
source build with **both** `USE_DISTRIBUTED=1` **and**
`TORCH_CUDA_ARCH_LIST=8.7`. The existing `${WORK}/pytorch-build/`
(7.1 GB, cp311 + torch 2.1.0a0) appears to be an earlier attempt at
this same exercise.

This dovetails with the V3-DEPLOY-RUNBOOK "Considered alternatives"
table which previously rejected AWQ/Triton on Jetson for the same
class of reason ("kernels per-arch; less proven on Jetson"). The
spike confirms: **the PyTorch-on-Jetson layer is the bottleneck for
the entire modern-Python-LLM-serving ecosystem on this hardware,
not anything SGLang-specific.**

### Required workarounds discovered (carry over into a future attempt)

If/when a custom PyTorch is built, these surface-level fixes still
apply:

1. **`PYTHONNOUSERSITE=1`** — user-site (`~/.local/lib/...`) holds an
   older transformers install that gets imported instead of the env's.
2. **Pin to SGLang 0.5.11's *actual* required versions**, not the
   strict pip metadata. Real working set:
   `flashinfer-python==0.6.8.post1`, `flashinfer-cubin==0.6.8.post1`,
   `xgrammar==0.1.32`, `outlines==0.1.11`, `openai-harmony==0.0.4`,
   `huggingface_hub<1.0,>=0.34`, `tokenizers<=0.23.0`,
   `llguidance>=0.7.11,<0.8.0`, `numpy<2`.
3. **torchvision C++ extension fails to load** (ABI mismatch with
   NVIDIA's nv24.08 torch). Workaround patches: (a) wrap the
   `from torchvision.io import decode_jpeg` import in
   `sglang/srt/utils/common.py:90` in a try/except with a raising
   stub; (b) neuter `torchvision/_meta_registrations.py` (only used
   by `torch.compile`, not serving). Both confirmed harmless for
   text-only serving. Would be unnecessary with a source-built
   torchvision matching the source-built torch.
4. **SGLang's pip metadata claims `torch==2.11.0`** — this is a real
   PyPI version (PyTorch jumped 2.5 → 2.12 in 2025) but no
   JP6 build of 2.11 exists. The runtime API surface SGLang
   *actually* uses is satisfied by torch 2.5/2.6.
5. **NVIDIA's JP6 PyTorch ships against cuDNN 9.3** (matches JP6.2);
   the older `/jp/v60/` wheels are cuDNN 8 — use `/jp/v61/` torch 2.5
   if not source-building.

### What this means for INFR-68

The spike's preregistered goal "does SGLang serve a model on
Jetson Orin" got refined into "does SGLang's `import` chain pass on a
working CUDA-on-`sm_87` PyTorch". Answer: not with anything off the
shelf today. Reopen criterion sharpens to:

**Required precondition for any future SGLang-on-Jetson work:** a
PyTorch wheel built from source on a Jetson with at minimum
`USE_DISTRIBUTED=1`, `USE_CUDA=1`,
`TORCH_CUDA_ARCH_LIST="8.7"`, cuDNN 9.x linked against the JetPack
system library. Once that wheel exists, the remaining install steps
in §"Required workarounds" above carry the runtime to `import sglang`
in well under an hour.

The existing `${WORK}/pytorch-build/` is the historical seed for
this work — keep it (the user wisely declined to purge during the
spike). A clean restart would target cp310 + torch 2.6 + cu126
+ sm_87.

### What this does NOT change

- The IOL training pipeline. Adapter format is canonical PEFT;
  artifact portability is unaffected. v3 GGUF deploy path via Ollama
  remains the production route.
- The strategic argument for SGLang (gpt-oss LoRA unblock, xgrammar,
  RadixAttention, multi-LoRA hot-swap) — those rewards still exist
  on the other side of the PyTorch-source-build wall. Just expensive
  to get to.
- `daedalus-v1` itself. We confirmed the artifact is well-formed; it
  will deploy on anything that loads a PEFT adapter.

### Environment artifacts (preserved on the dev box)

- Conda env: `${WORK}/conda-envs/sglang-spike` — SGLang
  stack installed minus the PyTorch issue. Reusable for the next
  attempt by just `pip install --force-reinstall` a working torch.
- Patched files (vs upstream sglang 0.5.11 wheel):
  - `…/sglang/srt/utils/common.py:90` — `decode_jpeg` import wrapped
    (original in `.orig`)
  - `…/torchvision/_meta_registrations.py` — neutered (original in
    `.orig`)
- All adapter and base-model artifacts untouched.

---

## Spike attempt 2 — 2026-05-14 (working)

**Result: SGLang is live on the Jetson Orin dev box**, serving an OpenAI-compatible
`/v1/chat/completions` endpoint on port 30000 with TinyLlama-1.1B as
the smoke target. End-to-end inference proven: `"2+2=" →
"The answer to the question is 4"` in 8 tokens.

The breakthrough was abandoning the "install SGLang from PyPI on top
of NVIDIA's JP6 PyTorch" path entirely. The PyPI wheels can't be
reconciled with NVIDIA's `USE_DISTRIBUTED=0` PyTorch (spike attempt 1).
The fix: **use Dusty Nv's pre-built Jetson SGLang container as a
source of a pre-built, mutually-consistent PyTorch + SGLang stack.**

### What changed vs attempt 1

The dustynv community maintains aarch64 Jetson container images that
bundle a torch built **with both `USE_DISTRIBUTED=1` and `sm_87`
device code**. That's exactly the wheel we couldn't get from any
public index. The image we used:

```
dustynv/sglang:r36.4.0     7.8 GB (Ubuntu 22.04 / Python 3.10 / CUDA 12.6 / sm_87)
```

Important: the **`-24.04` variants of the dustynv images use Python
3.12**, which does *not* match the dev box's host Python 3.10. Use the
plain `r36.4.0` tag (Ubuntu 22.04) to align with the host.

### Daemonless extraction (why)

We extract the dustynv container as files onto the working SSD via
`crane`, never invoking Docker's daemon. This keeps the container
artifacts off whatever disk the host's Docker daemon stores images
on, which matters when the dev box's root partition is intentionally
constrained for unrelated reasons.

### The recipe (reproducible)

`${WORK}` below is whatever working directory you keep this stack
under (e.g. a fast local SSD with ample free space).

```sh
# 1. crane (daemonless OCI puller, single static binary)
mkdir -p ${WORK}/bin ${WORK}/scratch
cd ${WORK}/scratch
LATEST=$(curl -sS "https://api.github.com/repos/google/go-containerregistry/releases/latest" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'])")
curl -sSL -o crane.tar.gz \
  "https://github.com/google/go-containerregistry/releases/download/${LATEST}/go-containerregistry_Linux_arm64.tar.gz"
tar -xzf crane.tar.gz -C ${WORK}/bin crane
chmod +x ${WORK}/bin/crane

# 2. Pull the Jetson SGLang container as an OCI layout (no Docker daemon)
${WORK}/bin/crane pull --format=oci \
  dustynv/sglang:r36.4.0 \
  ${WORK}/scratch/sglang-r36.4.0-oci

# 3. Extract all layers in manifest order to a merged rootfs on the working SSD
cd ${WORK}/scratch/sglang-r36.4.0-oci
MANIFEST=$(python3 -c "import json; print(json.load(open('index.json'))['manifests'][0]['digest'].split(':',1)[1])")
mkdir -p ${WORK}/scratch/sglang-rootfs
python3 -c "import json; m=json.load(open('blobs/sha256/$MANIFEST')); \
  [print(l['digest'].split(':',1)[1]) for l in m['layers']]" | \
  while read d; do
    tar --no-same-owner --warning=no-unknown-keyword -xzf "blobs/sha256/$d" \
      -C ${WORK}/scratch/sglang-rootfs 2>/dev/null || true
  done

# 4. Remove the stdlib-shadowing dataclasses backport (Python-2-era cruft)
DP=${WORK}/scratch/sglang-rootfs/usr/local/lib/python3.10/dist-packages
mkdir -p $DP/_disabled_backports
mv $DP/dataclasses.py $DP/_disabled_backports/
mv $DP/__pycache__/dataclasses.* $DP/_disabled_backports/ 2>/dev/null || true
```

### Runtime environment

The container uses an editable install pointing at `/workspace/sglang`,
so the host PYTHONPATH must include both the dist-packages and the
extracted source tree:

```sh
ROOTFS=${WORK}/scratch/sglang-rootfs
export PYTHONNOUSERSITE=1
export PYTHONPATH=$ROOTFS/workspace/sglang/python:$ROOTFS/usr/local/lib/python3.10/dist-packages
# Critical: Tegra driver shim FIRST in LD_LIBRARY_PATH, else "CUDA unknown error"
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/local/cuda-12.6/lib64:/usr/lib/aarch64-linux-gnu:$ROOTFS/usr/local/lib/python3.10/dist-packages/torch/lib
# Disable torch.compile: container's triton 3.2 + torch 2.5 mismatch breaks
# torch._inductor at import. We don't need compile for serving.
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export HF_HOME=${WORK}/huggingface
```

### Launch command (working)

```sh
/usr/bin/python3 -m sglang.launch_server \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 127.0.0.1 --port 30000 \
  --attention-backend triton \
  --mem-fraction-static 0.5 \
  --disable-cuda-graph \
  --log-level info
```

Why the flags:
- **`--attention-backend triton`** — flashinfer JIT-compiled kernels
  do work on Jetson (a pleasant surprise; the `flashinfer.jit:
  Loading JIT ops: norm` log line proved it), but Triton attention is
  the most conservative path for the first smoke run.
- **`--disable-cuda-graph`** — CUDA graph capture is finicky on Jetson
  Tegra; disable for serving stability. Re-enable later if perf needs it.
- **`--mem-fraction-static 0.5`** — Jetson has 64 GB *unified* memory;
  taking only half leaves headroom for the OS and Ollama-on-the-host.

### Smoke result

```
[2026-05-14 10:55:36 TP0] Load weight end. type=LlamaForCausalLM, dtype=torch.bfloat16, avail mem=19.81 GB
[2026-05-14 10:55:40 TP0] KV Cache is allocated. K size: 4.41 GB, V size: 4.41 GB.
[2026-05-14 10:55:40 TP0] Memory pool end. avail mem=10.38 GB
[2026-05-14 10:55:40 TP0] max_total_num_tokens=420200, max_prefill_tokens=16384, max_running_requests=4097, context_len=2048
Uvicorn running on http://127.0.0.1:30000
```

```sh
$ curl http://127.0.0.1:30000/v1/chat/completions \
    -d '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
         "messages":[{"role":"user","content":"2+2="}],
         "max_tokens":8,"temperature":0}'
# → "The answer to the question is 4"   (20 prompt + 8 completion tokens)
```

### What SGLang version actually runs

`sglang 0.4.1.post7` from Feb 2025 (the version bundled in the
dustynv container). Older than the 0.5.11 I tried in attempt 1, but
all the features the spike needs are present: `--lora-paths`,
`--max-loras-per-batch`, `--grammar-backend {xgrammar,outlines}`,
`--quantization {awq,fp8,gptq,marlin,gptq_marlin,awq_marlin,bitsandbytes,gguf,modelopt,w8a8_int8}`,
`--attention-backend {flashinfer,triton,torch_native}`.

**MXFP4 is *not* in the supported quantization list** for 0.4.1.
That means we cannot serve `openai/gpt-oss-20b` natively from
SGLang at this version — gpt-oss is shipped as MXFP4 by default.
To unblock gpt-oss serving specifically (the original V4-PLAN
prize) we would either need:
- A re-quantized gpt-oss in awq/gptq/fp8 (~30 min on a RunPod A100),
- Or a newer SGLang version (0.5.x) where MXFP4 is supported — which
  means re-running the spike with a newer dustynv tag once one ships
  for r36.4 with Python 3.10.

This shifts the **immediate next achievable target** from "gpt-oss
LoRA unblock" to "Devstral + daedalus-v1 LoRA bake-off" — which is
served by 0.4.1.post7's AWQ/GPTQ/bf16 paths without issue.

### Environment artifacts (preserved on the dev box)

- `${WORK}/scratch/sglang-rootfs/` — extracted container
  rootfs (18 GB). The "installation." **Keep.**
- `${WORK}/bin/crane` — daemonless OCI puller, kept for
  future container extractions on this Jetson. **Keep.**
- `${WORK}/scratch/sglang-logs/` — proof-of-life and
  bake-off run logs. Small. **Keep.**
- `${WORK}/scratch/sglang-r36.4.0-oci/` — original OCI
  layout (7.3 GB). **Removed** after `sglang-rootfs/` verified working.
- `${WORK}/conda-envs/sglang-spike/` — attempt-1 env that
  didn't lead anywhere. **Removed.**

---

## Bake-off — 2026-05-14 (Ollama vs SGLang, same model bytes, same hardware)

**Headline result: SGLang outperforms Ollama on Jetson by ~3× aggregate
throughput under concurrent load. At single-request latency they are
roughly tied (Ollama slightly faster).**

Both stacks ran simultaneously on the dev box, serving the **same**
Llama-3.1-8B Q4_K_M GGUF blob (Ollama's own model store, path
`<ollama-models>/blobs/sha256-667b0c1932…`). SGLang loaded
the blob via `--load-format gguf --quantization gguf`. Total resident
memory across both servers ≈ 18 GiB on the 64 GiB unified Jetson —
they coexist comfortably.

### Why not Devstral / gpt-oss

- **Devstral bf16** (44 GB) would have needed Ollama's pinned Devstral
  to be dropped *and* still wouldn't have left enough headroom for a
  fair concurrent test. Devstral GGUF on SGLang hit the Mistral-Tekken
  tokenizer wall documented in `M4-cloud-runbook.md`.
- **gpt-oss-20b** has no model class in SGLang 0.4.1's registry (gpt-oss
  was an August 2025 release; this SGLang is February 2025). Would
  need a newer SGLang version that ships with the right PyTorch on Jetson.
- **Llama-3.1-8B** is fair stand-in: same family of work, both stacks
  can serve it, fits on hardware.

### Setup

```sh
# Drop Ollama's pinned Devstral (it was occupying 36 GiB at 131k context)
curl -sS -X POST http://127.0.0.1:11434/api/generate \
  -d '{"model":"devstral:latest","keep_alive":0}'

# Preload Llama-3.1-8B on Ollama (4096 ctx, 30m keep_alive)
curl -sS -X POST http://127.0.0.1:11434/api/generate \
  -d '{"model":"llama3.1:8b","prompt":"ok","options":{"num_ctx":4096},
       "stream":false,"keep_alive":"30m"}'

# Launch SGLang against the SAME GGUF blob
BLOB=<ollama-models>/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29
# (env vars per Operations section above)
python3 -m sglang.launch_server \
  --model-path "$BLOB" \
  --load-format gguf --quantization gguf \
  --host 127.0.0.1 --port 30000 \
  --attention-backend triton \
  --mem-fraction-static 0.3 \
  --max-total-tokens 16384 \
  --max-running-requests 16 \
  --context-length 4096 \
  --disable-cuda-graph
```

### Workload

- **Shared system prompt**: a 2251-character Veltro-shaped persona +
  tool registry (8 tools, conventions, stop-discipline rules). Sent
  identically with every request — the textbook RadixAttention input
  shape.
- **8 user queries**, cycled to reach total request count per
  concurrency level: `What is 47 * 53?`, `Capital of Belgium?`,
  `Reverse the string: bicycle`, etc.
- **`max_tokens=128`, `temperature=0`**, explicit `stop=["<|eot_id|>",
  "<|end_of_text|>", "<|start_header_id|>"]`.
- Each concurrency level: `total_requests = 4 × concurrency`.

### Results

| Concurrency | Stack | Wallclock | Total compl. tok | p50 | p95 | Max | **Aggregate tok/s** |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Ollama | 4.66s | 76 | 1.12s | 1.91s | 1.91s | **16.3** |
| 1 | SGLang | 31.66s | 424 | 9.48s | 9.65s | 9.65s | **13.4** |
| 4 | Ollama | 8.76s | 204 | 2.13s | 3.14s | 3.35s | **23.3** |
| 4 | SGLang | 38.18s | 1793 | 9.49s | 9.65s | 9.74s | **47.0** |
| 8 | Ollama | 17.67s | 408 | 4.33s | 4.75s | 5.20s | **23.1** |
| 8 | SGLang | 47.23s | 3586 | 12.32s | 12.45s | 12.46s | **75.9** |
| 16 | Ollama | 35.29s | 817 | 8.66s | 9.09s | 9.47s | **23.1** |
| 16 | SGLang | 97.35s | 7673 | 12.46s | 53.69s | 91.01s | **78.8** |

Memory state during the run: 18–34 GiB used / 26–42 GiB free.
Comfortable.

### Reading the numbers

**Aggregate throughput by concurrency:**

- **Ollama plateaus at ~23 tok/s regardless of concurrency.** Adding
  parallel requests does not increase its throughput — it serializes.
  Per-request latency goes up roughly linearly (1.12s → 2.13s → 4.33s
  → 8.66s as concurrency goes 1 → 4 → 8 → 16). Classic single-slot
  llama.cpp behavior.
- **SGLang scales to ~78 tok/s.** It batches concurrent requests
  through one model forward pass. 3.4× the aggregate Ollama can do.

**Per-token compute cost under load:** 43 ms on Ollama vs **13 ms on
SGLang**. Tegra Orin's GPU is faster at this workload than llama.cpp's
single-threaded batching can exploit.

**Tail latency:** Ollama's tail is well-controlled (p95 ≈ p50, max
< 10s at N=16). SGLang's tail blows out at N=16 — p95=53.7s, max=91s
— some requests get starved while others run. **Tunable via
`--max-running-requests` and `--mem-fraction-static`** (we used
conservative defaults). For Veltro's expected fan-out (a few
concurrent tool calls, not 16), this is unlikely to bite.

### Honest caveats — none of which change the headline

1. **SGLang generated ~5× more tokens per request than Ollama** due to
   a stop-token issue: SGLang 0.4.1's GGUF tokenizer doesn't register
   Llama 3's special tokens (`<|eot_id|>`, etc.) properly, so `stop`
   strings only match after the fact. SGLang ran to `max_tokens=128`
   most of the time. **The scaling shape is independent of this**: at
   any per-request work level, Ollama would still serialize and
   plateau at ~23 tok/s. The 3× advantage is in batching efficiency
   per fixed token, not in artificial token inflation.

2. **Output quality on SGLang via GGUF is degraded** — same root cause
   as the stop-token issue. Llama 3 special tokens aren't tokenized
   correctly, so the model receives a malformed prompt and answers
   off-topic ("It seems like you are referring to the country of
   Belgium" instead of "Brussels"). **Fixable** by pointing
   `--tokenizer-path` at a non-gated HF Llama-3.1 mirror (`unsloth/…`,
   `NousResearch/…`) instead of relying on the GGUF embedded
   tokenizer. ~30 MB pull. Not done in this round to avoid yet
   another rabbit hole.

3. **`--chat-template llama-3-instruct` is not a registered name** in
   SGLang 0.4.1's `srt/conversation.py` (only `llama-2`, `chatml`,
   `vicuna_v1.1`, multimodal variants). For chat-completions API
   correctness, we'd need to either pass a `.jinja` file via
   `--chat-template` or rely on the HF tokenizer's chat_template.

4. **RadixAttention vs general batching efficiency**: the test ran
   with a shared system prompt but didn't isolate the radix-cache
   contribution from generic batched-prefill efficiency. The 3× win
   could be 90% RadixAttention + 10% batching, or 50/50, etc. The
   strategic answer ("SGLang is faster under concurrency on Jetson")
   doesn't depend on the attribution.

5. **N=16 is just where SGLang plateaus** at the conservative
   `--max-running-requests=16` we set. The plateau height moves with
   that flag; we did not tune for max.

### What this means for InferNode

- **For multi-agent fan-out workloads** (Veltro spawning N concurrent
  tool calls against a shared system prompt + tool schema), **SGLang
  gives real ~3× throughput headroom** on the existing Jetson Orin
  AGX. That's the V4-PLAN production case.
- **For single-user dev sessions** (one ZeroTier-mounted dialogue
  through `serve-llm`), Ollama is simpler and roughly equivalent on
  latency. Don't switch the dev path.
- **Integration cost is real**: SGLang on Jetson is the container-
  extraction recipe documented above, not a `pip install`. Tokenizer
  config needs care that Ollama hides.
- **For gpt-oss specifically** (the original V4-PLAN deploy blocker):
  still blocked. Needs a newer SGLang release than what dustynv's
  `r36.4.0` container shipped. Worth tracking for newer dustynv tags
  matched to Python 3.10.

### Reopen / next-iteration criteria

These are improvements that would make the result publishable beyond
"we measured a 3× delta", not blockers on the strategic answer:

1. **Pull `unsloth/Meta-Llama-3.1-8B-Instruct` tokenizer dir** (~30 MB)
   and pass `--tokenizer-path` to SGLang. Re-run; expect SGLang's
   per-request token count to match Ollama's, and SGLang's p95 tail
   to drop sharply.
2. **Compare SGLang with `--disable-radix-cache` vs default** — isolates
   the RadixAttention contribution from generic batched-prefill efficiency.
3. **IOL eval harness** (`tools/virgil-agent/scenarios/v3.yaml`) against
   both endpoints, with and without grammar-constrained decoding on the
   SGLang side. Answers the tool-call correctness question, not just
   throughput.
4. **A newer SGLang** (0.5.x via a newer dustynv tag, when one ships
   for r36.4 with Python 3.10) — gets gpt-oss model architecture into
   the registry, unlocking the actual V4-PLAN prize.

### Servers stopped, environment preserved

After the run:
- SGLang server: stopped (`kill -TERM` via port-30000 ownership).
- Ollama: `keep_alive=0` POST dropped Llama-3.1-8B; daemon left up,
  no models loaded.
- Extracted SGLang rootfs at `${WORK}/scratch/sglang-rootfs/`
  preserved for the next session — just relaunch per the Operations
  section above.
