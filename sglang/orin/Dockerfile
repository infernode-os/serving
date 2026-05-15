# SGLang for Jetson Orin AGX (sm_87, JetPack 6.x, CUDA 12.9)
#
# **v1 ships with dustynv's bundled SGLang 0.4.9** — no upgrade. This is a
# deliberate rollback from the 0.5.4 attempt (commit a5b9b84 → e3bc786):
# SGLang 0.5.x strictly pins sgl-kernel 0.3.16.post3, but that wheel ships
# only sm_90/sm_100 binaries. Cross-arch fallback at *import* time succeeds,
# but kernel calls (rmsnorm, etc.) fail at *runtime* with "no kernel image
# available for execution on the device". Building sgl-kernel 0.3.16.post3
# from source for sm_87 is blocked by an upstream broken deepgemm commit
# SHA in sgl-kernel's CMakeLists.txt (FetchContent_Populate fails). See
# INFR-92 for the path to a working sgl-kernel-from-source build.
#
# Trade-off documented: SGLang 0.4.9 has no `gpt_oss.py` model arch, so
# lucibridge's gpt-oss routing falls back to Ollama (per
# `runbooks/lucibridge-routing.md` fallback semantics) until INFR-92 lands.
# All other model architectures supported by SGLang 0.4.9 work normally —
# this image actually *serves requests* on Orin, which the 0.5.4 attempts
# did not.
#
# Recipe:
#   1. FROM dustynv/sglang:r36.4-cu129-24.04 (torch 2.8.0+cu129, sgl-kernel
#      0.2.3 BUILT FOR sm_87 by dustynv, triton 3.4.0, vllm 0.9.3, sglang
#      0.4.9 — all working on Orin out of the box).
#   2. One-line sed patch on vllm/transformers_utils/configs/ovis.py to add
#      exist_ok=True on the `aimv2` registration. vllm 0.9.3 + transformers
#      4.53.1 collide because transformers 4.53+ has aimv2 natively.
#   3. Bake Llama-3 / Llama-3.1 tokenizer dirs at /opt/tokenizers/ (INFR-78).
#   4. Smoke test + on-hardware validator scripts.

ARG BASE_IMAGE=dustynv/sglang:r36.4-cu129-24.04
FROM ${BASE_IMAGE}

ARG TOKENIZER_DIR=/opt/tokenizers \
    VENV=/opt/venv

# vllm 0.9.3 + transformers 4.53.1 aimv2 registration collision workaround.
# Idempotent — second run is a no-op.
RUN sed -i \
        's|AutoConfig.register("aimv2", AIMv2Config)|AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)|' \
        ${VENV}/lib/python3.12/site-packages/vllm/transformers_utils/configs/ovis.py \
 && grep -q 'exist_ok=True' \
        ${VENV}/lib/python3.12/site-packages/vllm/transformers_utils/configs/ovis.py

# dustynv's sglang 0.4.9 ships a SyntaxError in srt/utils.py: an orphan
# `except FileNotFoundError:` block left behind by a refactor of
# `get_nvgpu_memory_capacity()`. Both code paths (Jetson + non-Jetson)
# already handle exceptions inside their own try-blocks, so the orphan
# is plain dead syntax. dustynv apparently didn't `python -m sglang.launch_server`
# before publishing the image. Strip the four orphan lines with a Python
# patch (sed across multi-line block is fragile). Idempotent.
RUN ${VENV}/bin/python3 - <<'PY'
import re, pathlib
p = pathlib.Path('/opt/venv/lib/python3.12/site-packages/sglang/srt/utils.py')
src = p.read_text()
# The orphan block sits after the inner try-except for the non-Jetson path
# closes (two excepts, both at 8-space indent), then there's blank lines,
# then the orphan `except` at 4-space indent. Match exactly that block.
pattern = re.compile(
    r'(\n\n)    except FileNotFoundError:\n'
    r'        raise RuntimeError\(\n'
    r'            "nvidia-smi not found\. Ensure NVIDIA drivers are installed and accessible\."\n'
    r'        \)\n',
    re.MULTILINE,
)
new, n = pattern.subn(r'\1', src)
if n == 0:
    # Either already patched, or upstream changed; be loud either way.
    if 'nvidia-smi not found. Ensure NVIDIA drivers are installed and accessible.' in src:
        raise SystemExit('FAIL: orphan-except patch did not match but error string still present')
    print('orphan-except patch: nothing to do (already clean)')
else:
    p.write_text(new)
    print(f'orphan-except patch: removed {n} block(s)')
import ast
ast.parse(new)  # syntactic validation
print('utils.py: parses cleanly')
PY

# Build-time guards: GPU-less CI runners limit us to metadata checks.
# Anything that transitively imports sgl_kernel runs `_load_architecture_specific_ops()`
# which probes for a CUDA device — fails import on the runner even when
# the wheel is correct. Deep on-hardware checks live in validate-on-hardware.sh.
COPY validate-on-hardware.sh /opt/sglang/validate-on-hardware.sh
RUN chmod +x /opt/sglang/validate-on-hardware.sh \
 && ${VENV}/bin/python3 -c "import torch; assert torch.version.cuda is not None, f'CPU-only torch: {torch.__version__}'; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)" \
 && ${VENV}/bin/python3 -c "import triton; print('triton:', triton.__version__)" \
 && ${VENV}/bin/python3 -c "import importlib.metadata as m; v=m.version('sglang'); print('sglang metadata:', v); assert v.startswith('0.4'), f'expected 0.4.x baseline (INFR-92 covers 0.5+ upgrade), got {v}'"

# Bake Llama-3 family tokenizer dirs (INFR-78).
COPY bake-tokenizers.sh /tmp/
RUN TOKENIZER_DIR=${TOKENIZER_DIR} /tmp/bake-tokenizers.sh

# Upstream smoke-test script preserved at the documented path.
COPY test.py /opt/sglang/test.py

LABEL org.opencontainers.image.source="https://github.com/infernode-os/serving"
LABEL org.opencontainers.image.description="SGLang 0.4.9 for Jetson Orin AGX (sm_87) on JetPack 6 / CUDA 12.9 — InferNode delta on dustynv/sglang. v1; gpt_oss support pending INFR-92."
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.infernode.base="${BASE_IMAGE}"
LABEL org.infernode.gpt_oss_supported="false"
LABEL org.infernode.followups="INFR-92"
