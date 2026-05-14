# SGLang for Jetson Orin AGX (sm_87, JetPack 6.x, CUDA 12.6)
#
# Based on dusty-nv/jetson-containers:packages/llm/sglang/Dockerfile
# (see ../LICENSE-UPSTREAM.md). Kept verbatim at ./Dockerfile.upstream
# for diff-against-upstream. This file is the standalone production
# build with the following deltas:
#
#   1. `transformers` install moved to a `pip install` (upstream chains
#      its own /tmp/transformers/install.sh from an earlier stage).
#   2. Tokenizer + chat-template bake step added at /opt/tokenizers/
#      (INFR-78).
#   3. Smoke-test script copied to /opt/sglang/test.py.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SGLANG_VERSION \
    SGLANG_VERSION_SPEC \
    IS_SBSA \
    FORCE_BUILD=off \
    TMP=/tmp/sglang \
    TOKENIZER_DIR=/opt/tokenizers \
    PIP_WHEEL_DIR=/opt/wheels

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        libnuma-dev \
        libsndfile1 \
        libsndfile1-dev \
        libprotobuf-dev \
        libsm6 \
        libxext6 \
        libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install.sh + build.sh are vendored verbatim from dusty-nv/jetson-containers and
# assume the framework's parent stages have already provided `uv` on PATH and
# exported PIP_WHEEL_DIR. We satisfy both prereqs at the Dockerfile layer so the
# vendored scripts stay untouched (clean diffs against upstream re-syncs).
ENV PIP_WHEEL_DIR=${PIP_WHEEL_DIR}
# dustynv/pytorch ships pip pre-configured for the JetPack PyPI mirror
# (pypi.jetson-ai-lab.dev/jp6/cu128/) which carries Jetson-built wheels
# for torch/triton/etc. with sm_87 device code, but does NOT carry `uv`.
# So: install uv from upstream PyPI (one-shot), then point uv at the
# JetPack mirror for everything else. Critical: `uv` ignores PIP_*
# env vars — it only reads UV_*. Without UV_INDEX_URL set, uv resolves
# `sglang[all]`'s transitive torch dep to the upstream CPU wheel and
# clobbers the base image's CUDA torch (smoke-tested: torch 2.9.1+cpu
# on a Tegra GPU, no CUDA visible inside the container).
RUN mkdir -p ${PIP_WHEEL_DIR} && \
    pip install --no-cache-dir --index-url https://pypi.org/simple uv

ENV UV_INDEX_URL=https://pypi.jetson-ai-lab.dev/jp6/cu128 \
    UV_EXTRA_INDEX_URL=https://pypi.org/simple \
    UV_INDEX_STRATEGY=unsafe-best-match

# Vendored SGLang install (pip-first; falls back to source build on failure).
COPY build.sh install.sh $TMP/
RUN $TMP/install.sh || $TMP/build.sh || touch $TMP/.build.failed
RUN if [ -f $TMP/.build.failed ]; then \
      echo "SGLANG ${SGLANG_VERSION} build failed!"; \
      exit 1; \
    fi

# Standalone replacement for upstream's chained /tmp/transformers/install.sh.
# SGLang declares its own transformers pin via setup.py; this just ensures the
# transitive install actually completed.
RUN python3 -c "import transformers; print('transformers', transformers.__version__)" \
    || pip install --no-cache-dir 'transformers>=4.45,<5'

# Bake Llama-3 family tokenizer + chat templates so --tokenizer-path can
# be set at launch time without a runtime HuggingFace pull. Adds ~60 MB
# (two families * ~30 MB) to the final image. (INFR-78.)
COPY bake-tokenizers.sh $TMP/
RUN TOKENIZER_DIR=${TOKENIZER_DIR} $TMP/bake-tokenizers.sh

# Smoke test, callable as `docker run --rm <image> python3 /opt/sglang/test.py`.
COPY test.py /opt/sglang/test.py

LABEL org.opencontainers.image.source="https://github.com/infernode-os/serving"
LABEL org.opencontainers.image.description="SGLang for Jetson Orin AGX (sm_87) — InferNode fork of dusty-nv/jetson-containers"
LABEL org.opencontainers.image.licenses="MIT"
