# InferNode serving

Container builds, deployment recipes, and routing configuration for
the LLM serving stack behind InferNode's `llmsrv` / `lucibridge` /
`serve-llm.sh`.

**Status: WIP.** Repo bootstrapped 2026-05-14 to productize findings
from INFR-68 (SGLang-on-Jetson spike). Read
[`docs/SGLANG-ADOPTION-NOTES.md`](docs/SGLANG-ADOPTION-NOTES.md) for
the full measured-results writeup and the working installation
recipe.

## Why a separate repo

- **`infernode-os/infernode`** is the OS / runtime (Limbo, emu,
  Veltro, lucifer). Docker/CI scaffolding for *external* serving
  stacks doesn't belong inside the OS source.
- **`pdfinn/infernode-os-llm`** is the training pipeline (corpus →
  harvest → train → eval → adapter). Its lifecycle moves with model
  training cycles.
- **This repo** owns the serving runtime that consumes IOL's
  adapters: Jetson-targeted container builds, lucibridge per-tool
  routing configs, deployment runbooks. Lifecycle is tied to JetPack
  releases / SGLang versions / hardware generations, not to model
  training cycles.

## Hardware targets

- **Jetson Orin AGX** (`sm_87`, JetPack 6.x): current production. The
  measured 3× concurrent-throughput advantage of SGLang over Ollama
  was characterized on this hardware.
- **Jetson Thor** (`sm_103`, JetPack 7.x): forward-looking. NVIDIA
  ships official NGC SGLang containers for Thor (per the
  [Run SGLang in Thor](https://forums.developer.nvidia.com/t/run-sglang-in-thor/348815)
  forum thread). Cleaner upstream than Orin's community-maintained
  path.

## Planned structure

```
serving/
├── README.md
├── LICENSE                    (MIT)
├── docs/
│   ├── SGLANG-ADOPTION-NOTES.md   working/measured notes (moved from IOL)
│   └── ...
├── sglang/                    fork of dusty-nv/jetson-containers/packages/llm/sglang/
│   ├── orin/                  pinned config for Orin (sm_87)
│   └── thor/                  pinned config for Thor (sm_103) — when we have a Thor box
├── runbooks/
│   ├── hephaestus-deploy.md
│   └── ...
└── .github/workflows/
    └── build-jetson-container.yml   self-hosted runner on Hephaestus
```

The `sglang/` subtree is intended to vendor the canonical
[`dusty-nv/jetson-containers`](https://github.com/dusty-nv/jetson-containers/tree/master/packages/llm/sglang)
recipe with our pinned SGLang version (≥ 0.5.x so `gpt_oss.py` is in
the model registry).

## Upstream sources we depend on / track

- **dusty-nv / jetson-containers** —
  https://github.com/dusty-nv/jetson-containers (the canonical
  Jetson container build framework; NVIDIA-DevRel-maintained, MIT
  licensed)
- **sgl-project / sglang** — https://github.com/sgl-project/sglang
  (upstream; lmsysorg's official images are SBSA — datacenter ARM,
  *not* Jetson Tegra)
- **NGC SGLang catalog** —
  https://catalog.ngc.nvidia.com/orgs/nvidia/containers/sglang
  (NVIDIA-published; recent tags target Thor sm_103)
- **dustynv/sglang on Docker Hub** —
  https://hub.docker.com/r/dustynv/sglang (pre-built artifacts of
  jetson-containers; community-maintained; what INFR-68 spike used —
  `r36.4.0` tag with SGLang 0.4.1)

## Working notes

See `docs/SGLANG-ADOPTION-NOTES.md` for:
- The spike attempts that didn't work and why (PyTorch wheel
  `USE_DISTRIBUTED` vs `sm_87` constraint)
- The working recipe (extract dustynv container via crane onto
  orin-ssd, host Python 3.10 with patched `LD_LIBRARY_PATH`)
- Measured bake-off results — SGLang vs Ollama on Llama-3.1-8B
  (TL;DR: SGLang scales to ~78 tok/s at N=8 concurrent vs Ollama's
  ~23 tok/s plateau; tied at single-user)
- Operations: where things live on Hephaestus, start/stop, verify
- Known gaps: SGLang 0.4.1's GGUF tokenizer doesn't recognize Llama
  3 special tokens (needs HF tokenizer dir); no `gpt_oss.py` in
  0.4.1's model registry (needs SGLang 0.5.x bump)

## Hephaestus disk policy (important for the build path)

Hephaestus is the Jetson Orin AGX dev box. Its root partition is
**deliberately constrained** to emulate a production single-disk
node (OS + TAK + NERVA via Docker + Ollama binary). The 916 GB
`/mnt/orin-ssd` is the dev indulgence; serving-spike artifacts live
there.

**Do not migrate Docker / containerd state from root to orin-ssd.**
That would move TAK/NERVA images onto the dev-only disk and break
the production emulation. The Jetson-container build needs to either
fit in root partition's residual space, or use a daemonless build
path (we did extraction-only via
[`crane`](https://github.com/google/go-containerregistry) on the
spike; expect to reuse).

## Tracking

Work in this repo is tracked under the **INFR Jira project's
"Productize SGLang serving" epic**. See INFR-68 (the original spike)
and its child tickets.
