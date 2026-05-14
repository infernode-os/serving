# sglang/ — vendored Jetson SGLang recipe

This subtree holds container recipes for running SGLang on NVIDIA Jetson
hardware, vendored from
[`dusty-nv/jetson-containers`](https://github.com/dusty-nv/jetson-containers)
(MIT-licensed, NVIDIA-DevRel-maintained) and adapted for InferNode's
production needs.

## Layout

```
sglang/
├── README.md                  this file
├── LICENSE-UPSTREAM.md        attribution for vendored dusty-nv recipe
├── orin/                      Jetson Orin AGX (sm_87, JetPack 6.x, CUDA 12.6)
│   ├── README.md
│   ├── config.py              OUR pinned SGLang version
│   ├── Dockerfile             vendored from upstream
│   ├── build.sh               vendored from upstream
│   ├── install.sh             vendored from upstream
│   └── test.py                vendored from upstream
└── thor/                      Jetson Thor (sm_103, JetPack 7.x, CUDA 13.x)
    ├── README.md
    └── Dockerfile             thin wrapper over NGC's nvcr.io/nvidia/sglang
```

## Why two variants

The two hardware generations need different base-image strategies:

* **Orin (`sm_87`)** has no NGC SGLang image (per INFR-74 investigation —
  NGC's SGLang line is CUDA-13-based, JP7-targeted). The fork-and-build
  path is the only one available. The Orin recipe is a full vendor of
  the dusty-nv jetson-containers SGLang package with our pinned version.

* **Thor (`sm_103`)** has NVIDIA's official NGC SGLang container
  (`nvcr.io/nvidia/sglang:25.10-py3` and later). Cleaner upstream than
  Orin's community-maintained chain — we use it as a base and add only
  what InferNode-specific bits we need (tokenizers, entrypoint
  conveniences).

## Vendoring decision

Straight copy with attribution, not git submodule or subtree-merge. Reasons:

1. We need to **diverge** from upstream's pinned version: upstream pins
   SGLang 0.5.11 with the explicit annotation "Compatible with CUDA 13
   (Spark and Thor)". JetPack 6.x ships CUDA 12.6 — that pin doesn't
   work for Orin. We need our own version pin (see `orin/config.py`).
2. The upstream recipe is **small** (≈ 130 LOC across 4 files plus
   README + test). Submodule overhead exceeds the merge-back cost.
3. We may want to apply Orin-specific patches (e.g. the chat-template
   / tokenizer fixes per INFR-78) without coordinating with upstream.

If upstream evolves in ways we care about, the re-sync is a manual
diff-and-merge against `LICENSE-UPSTREAM.md`'s recorded commit SHA.
The vendored files at the time of copy are an exact snapshot — the
diff is therefore easy to compute.

## Upstream source

* Repo: <https://github.com/dusty-nv/jetson-containers>
* Path: `packages/llm/sglang/`
* Vendored from: `master` branch as of 2026-05-14
* Upstream commit at vendoring: see `LICENSE-UPSTREAM.md`
* Upstream license: MIT (compatible with this repo's MIT)

## What changed vs upstream

| File | Status |
|---|---|
| `orin/config.py` | **Modified** — pinned to a 0.5.x release compatible with CUDA 12.6 (see `orin/README.md`) |
| `orin/Dockerfile.upstream` | Verbatim — kept for diff against upstream re-syncs |
| `orin/Dockerfile` | **Modified** — standalone build (drops chained `/tmp/transformers/install.sh`, adds tokenizer bake step per INFR-78) |
| `orin/build.sh` | Verbatim |
| `orin/install.sh` | Verbatim |
| `orin/test.py` | Verbatim |
| `orin/bake-tokenizers.sh` | InferNode-authored — pulls non-gated Llama-3 tokenizer dirs (INFR-78) |
| `thor/Dockerfile` | InferNode-authored — wraps NGC `nvcr.io/nvidia/sglang` |
| `thor/test.py` | Verbatim copy of `orin/test.py` (docker-context boundary) |

When you re-sync from upstream, diff the `orin/` non-config files
against the upstream snapshot at that point; `config.py` is intentionally
divergent and should not be auto-overwritten.
