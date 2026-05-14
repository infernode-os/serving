# SGLang for Jetson Orin AGX (sm_87, JetPack 6.x, CUDA 12.9)
#
# **Rewritten 2026-05-14** — the original dusty-nv/jetson-containers install.sh
# path could not produce a working image: every SGLang 0.5.x version strictly
# pins `torch==2.8.0`, and torch 2.8 + cu128 + sm_87 + cp312 doesn't exist as
# a public wheel anywhere (upstream PyPI tops out at cp310 aarch64; NVIDIA's
# JP redist tops out at torch 2.4; the JetPack mirror only has cp310). The
# only place that combination existed was inside dustynv's pre-built SGLang
# image — but at SGLang 0.4.9 with no gpt_oss.py. INFR-91 / INFR-77.
#
# Recipe (delta-upgrade on top of dustynv's working CUDA SGLang base):
#   1. FROM dustynv/sglang:r36.4-cu129-24.04 (torch 2.8.0+cu129, sgl-kernel,
#      triton 3.4.0, flashinfer, vllm 0.9.3 — all built for sm_87)
#   2. Uninstall stale SGLang 0.4.9 (no gpt_oss.py) and sgl-kernel 0.2.3
#   3. Delta-install SGLang 0.5.4 + sgl-kernel 0.3.16.post3 with
#      --prerelease=allow (transitive flashinfer pin is a pre-release)
#   4. Patch vllm 0.9.3's ovis.py to register `aimv2` with exist_ok=True
#      — vllm tries to register a type that transformers 4.57.1 has natively,
#      causing import to die without the patch
#   5. Bake Llama-3 tokenizer dirs at /opt/tokenizers/ (INFR-78)
#   6. Copy upstream smoke test to /opt/sglang/test.py

ARG BASE_IMAGE=dustynv/sglang:r36.4-cu129-24.04
FROM ${BASE_IMAGE}

ARG SGLANG_VERSION=0.5.4 \
    SGL_KERNEL_VERSION=0.3.16.post3 \
    TOKENIZER_DIR=/opt/tokenizers \
    VENV=/opt/venv

# Step 1+2+3: clean delta-upgrade from dustynv's bundled SGLang 0.4.9 to 0.5.4.
# torch 2.8.0+cu129 in the base satisfies sglang 0.5.4's strict pin so uv
# doesn't try to upgrade it. --prerelease=allow needed for the transitive
# flashinfer-python==0.4.0rc3 pin.
RUN ${VENV}/bin/pip uninstall -y sglang sgl-kernel sgl-router \
 && ${VENV}/bin/uv pip install --python ${VENV}/bin/python --prerelease=allow \
        "sglang==${SGLANG_VERSION}" \
        "sgl-kernel==${SGL_KERNEL_VERSION}"

# Step 4: vllm 0.9.3 + transformers 4.57.1 collision workaround.
# vllm's ovis.py registers the `aimv2` model type, but transformers 4.57+
# has it natively, so the registration raises ValueError on import. Fix:
# pass exist_ok=True. One-line sed; rerun-safe (idempotent).
RUN sed -i \
        's|AutoConfig.register("aimv2", AIMv2Config)|AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)|' \
        ${VENV}/lib/python3.12/site-packages/vllm/transformers_utils/configs/ovis.py \
 && grep -q 'exist_ok=True' \
        ${VENV}/lib/python3.12/site-packages/vllm/transformers_utils/configs/ovis.py

# Build-time guards: if any of these fail, fail the build instead of
# publishing a broken image. INFR-91's CPU-torch regression would have been
# caught by the first one.
RUN ${VENV}/bin/python3 -c "import torch; assert torch.version.cuda is not None, f'CPU-only torch: {torch.__version__}'; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)" \
 && ${VENV}/bin/python3 -c "import triton; print('triton:', triton.__version__)" \
 && ${VENV}/bin/python3 -c "import sgl_kernel; print('sgl_kernel: OK')" \
 && ${VENV}/bin/python3 -c "import sglang; print('sglang:', sglang.__version__)" \
 && ${VENV}/bin/python3 -c "import sglang.srt.models.gpt_oss as m; print('gpt_oss:', m.__file__)" \
 && ${VENV}/bin/python3 -c "from sglang.srt.entrypoints.http_server import launch_server; print('launch_server: OK')"

# Step 5: Bake Llama-3 family tokenizer dirs (INFR-78).
COPY bake-tokenizers.sh /tmp/
RUN TOKENIZER_DIR=${TOKENIZER_DIR} /tmp/bake-tokenizers.sh

# Step 6: Smoke test script.
COPY test.py /opt/sglang/test.py

LABEL org.opencontainers.image.source="https://github.com/infernode-os/serving"
LABEL org.opencontainers.image.description="SGLang ${SGLANG_VERSION} for Jetson Orin AGX (sm_87) on JetPack 6 / CUDA 12.9 — InferNode delta on dustynv/sglang"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.infernode.sglang.version="${SGLANG_VERSION}"
LABEL org.infernode.sgl_kernel.version="${SGL_KERNEL_VERSION}"
LABEL org.infernode.base="${BASE_IMAGE}"
