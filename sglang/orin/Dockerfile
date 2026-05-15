# SGLang for Jetson Orin AGX (sm_87, JetPack 6.x, CUDA 12.6)
#
# **v1 ships SGLang 0.4.1.post7 on the spike-proven dustynv base.** This is
# the bundle that served TinyLlama and ran the Ollama-vs-SGLang bake-off on
# 2026-05-14 (docs/SGLANG-ADOPTION-NOTES.md §"Spike attempt 2"), measured at
# ~3× aggregate throughput vs Ollama under concurrent load. The bundle is
# self-consistent: dustynv built torch with USE_DISTRIBUTED=1 *and* sm_87
# device code — exactly the wheel that no public index ships.
#
# v1 trade-offs accepted:
#  * No gpt_oss.py model arch in 0.4.1 (added Aug 2025). lucibridge falls
#    back to Ollama for gpt-oss routing per runbooks/lucibridge-routing.md.
#  * No MXFP4 quantisation in 0.4.1's supported list.
#  * Both deferred to INFR-92 (newer SGLang on a newer dustynv tag).
#
# v0.5.x attempts (INFR-91/INFR-92) and a delta upgrade on dustynv's cu129
# image are documented in those tickets; both surfaced enough untested
# upstream bugs that we ship the spike-proven baseline first and pursue the
# upgrade as a follow-up against a working reference.
#
# Recipe:
#   1. FROM dustynv/sglang:r36.4.0 (Ubuntu 22.04 / Python 3.10 / torch 2.5
#      cu126 / triton 3.2.0 / sgl-kernel sm_87 / sglang 0.4.1.post7 editable
#      at /workspace/sglang).
#   2. Remove the dataclasses backport that shadows the stdlib (spike fix —
#      dustynv bundles a Python-2-era dataclasses.py in dist-packages).
#   3. Bake Llama-3 / Llama-3.1 tokenizer dirs at /opt/tokenizers/ (INFR-78).
#   4. Build-time guards (CI runner has no GPU) + on-hardware validator.

ARG BASE_IMAGE=dustynv/sglang:r36.4.0
FROM ${BASE_IMAGE}

ARG TOKENIZER_DIR=/opt/tokenizers

# torch.compile + CUDA-graph capture are unreliable on Jetson Tegra (spike
# note in SGLANG-ADOPTION-NOTES.md §"Runtime environment"). We don't need
# compile for serving; bake the disables into the image so any future
# wrapper can't forget them. Runtime can still override.
ENV TORCHDYNAMO_DISABLE=1 \
    TORCH_COMPILE_DISABLE=1 \
    PYTHONNOUSERSITE=1

# Remove the dataclasses backport that shadows Python 3's stdlib module.
# dustynv bundles dataclasses-0.x (a Python-2.7 backport) in dist-packages;
# it sits ahead of the stdlib on sys.path and breaks modern code. The
# spike's §"Required workarounds" extracts it from the rootfs; we do the
# same inside the image. Idempotent.
RUN DP=/usr/local/lib/python3.10/dist-packages \
 && mkdir -p ${DP}/_disabled_backports \
 && if [ -f ${DP}/dataclasses.py ]; then \
        mv ${DP}/dataclasses.py ${DP}/_disabled_backports/ ; \
        mv ${DP}/__pycache__/dataclasses.* ${DP}/_disabled_backports/ 2>/dev/null || true ; \
        echo "moved dataclasses backport" ; \
    else \
        echo "dataclasses backport already absent (idempotent re-run)" ; \
    fi \
 && python3 -c "import dataclasses, sys; \
                p = dataclasses.__file__ or ''; \
                assert 'dist-packages' not in p and 'site-packages' not in p, \
                    f'dataclasses still resolves outside stdlib: {p}'; \
                print('dataclasses resolves to stdlib:', p)"

# Build-time guards: CI runners have no GPU, so this is metadata-only.
# Real on-hardware verification lives in validate-on-hardware.sh and runs
# post-pull on Hephaestus / a field Orin.
COPY validate-on-hardware.sh /opt/sglang/validate-on-hardware.sh
COPY test.py /opt/sglang/test.py
RUN chmod +x /opt/sglang/validate-on-hardware.sh \
 && python3 -c "import torch; assert torch.version.cuda is not None, \
                f'CPU-only torch: {torch.__version__}'; \
                print('torch:', torch.__version__, 'cuda:', torch.version.cuda)" \
 && python3 -c "import triton; print('triton:', triton.__version__)" \
 && python3 -c "import importlib.metadata as m; \
                v = m.version('sglang'); \
                print('sglang:', v); \
                assert v == '0.4.1.post7', \
                    f'expected spike-pinned sglang 0.4.1.post7 (INFR-92 covers upgrade), got {v}'"

# Bake Llama-3 family tokenizer dirs (INFR-78).
COPY bake-tokenizers.sh /tmp/
RUN TOKENIZER_DIR=${TOKENIZER_DIR} /tmp/bake-tokenizers.sh

# BASE_IMAGE referenced in a label must be re-declared in this stage —
# ARGs from before the first FROM are out of scope by default.
ARG BASE_IMAGE
LABEL org.opencontainers.image.source="https://github.com/infernode-os/serving"
LABEL org.opencontainers.image.description="SGLang 0.4.1.post7 for Jetson Orin AGX (sm_87) on JetPack 6 / CUDA 12.6 — spike-proven v1 on dustynv/sglang:r36.4.0. gpt-oss + MXFP4 pending INFR-92."
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.infernode.base="${BASE_IMAGE}"
LABEL org.infernode.sglang_version="0.4.1.post7"
LABEL org.infernode.gpt_oss_supported="false"
LABEL org.infernode.mxfp4_supported="false"
LABEL org.infernode.followups="INFR-92"
