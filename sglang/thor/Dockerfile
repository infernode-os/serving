# Jetson Thor (sm_103, JetPack 7, CUDA 13) SGLang container.
#
# This is a thin wrapper over NVIDIA's official NGC SGLang image. The
# heavy lifting (PyTorch with sm_103 device code, sgl-kernel, flashinfer,
# xgrammar) is all done by NVIDIA in the base image. We add only what
# InferNode needs on top.
#
# Build: see sglang/thor/README.md or .github/workflows/build-sglang.yml.

ARG NGC_SGLANG_TAG=25.10-py3
FROM nvcr.io/nvidia/sglang:${NGC_SGLANG_TAG}

LABEL org.opencontainers.image.source="https://github.com/infernode-os/serving"
LABEL org.opencontainers.image.description="SGLang for Jetson Thor (sm_103) — InferNode overlay on NGC base"
LABEL org.opencontainers.image.licenses="MIT"

# Smoke test mirrored from sglang/orin/test.py so the same verification
# script works against either variant. (Duplicated rather than symlinked
# because docker build context can't reach outside this directory.)
COPY test.py /opt/sglang/test.py

# No CMD/ENTRYPOINT override — preserve NGC's defaults so callers
# pass the launch args explicitly (see runbooks/deploy.md).
