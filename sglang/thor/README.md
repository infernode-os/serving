# sglang/thor/ — Jetson Thor (sm_103)

Container recipe for SGLang on Jetson Thor. Forward-looking — we
don't have a Thor box yet, but the recipe stays parallel to Orin's so
that the deploy story doesn't need a rewrite when one arrives.

## Target

| Property | Value |
|---|---|
| GPU | Blackwell Tegra (`sm_103`) |
| JetPack | 7.x |
| CUDA | 13.0+ |
| Base | `nvcr.io/nvidia/sglang:25.10-py3` (NGC) |

## Why this is shorter than `orin/`

NVIDIA ships an **official** NGC SGLang container line starting at
`25.10-py3` (October 2025) that explicitly targets Jetson Thor. See
the INFR-74 investigation comment for the full enumeration. The Thor
recipe therefore doesn't fork dusty-nv/jetson-containers — it just
pulls NGC's image, version-pins, and layers any InferNode-specific
conveniences on top.

This is a deliberate asymmetry with `orin/`: NGC's SGLang line is
CUDA-13-based and JP7-targeted, so it can't run on Orin/JP6 — but
that's exactly what we want for Thor.

## Pinned base

`nvcr.io/nvidia/sglang:25.10-py3` — first NGC release with Jetson
Thor support. Bump as newer NGC tags ship (`25.11-py3`, `26.02-py3`,
etc.) once we have a Thor box and can verify each.

## Known issues to track

Per the [NGC SGLang 25.10 release notes](https://docs.nvidia.com/deeplearning/frameworks/sglang-release-notes/rel-25-10.html):

* **`gpt-oss` family models cannot run on DGX Spark and Jetson Thor
  due to an OpenAI Triton issue.** This blocks the V4-PLAN gpt-oss
  prize on Thor specifically. Track NGC release notes for resolution;
  Orin's recipe (with a JP6/CUDA-12.6 Triton stack) is the workaround
  in the meantime.

## Build

CI matrix-builds both variants on `ubuntu-24.04-arm`; the Thor variant
is a `FROM nvcr.io/nvidia/sglang:25.10-py3` + thin overlay, so the
build is mostly a re-tag with our overlay scripts. See
`.github/workflows/build-sglang.yml`.

NGC auth is required for the pull (free NGC account works; the CI
workflow expects `NGC_API_KEY` as a repo secret).

## Manual build on a Thor host (when one exists)

```sh
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
docker build -t serving-sglang:thor-local sglang/thor/
```
