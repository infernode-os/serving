#!/usr/bin/env bash
# Run after pulling the published image to a Jetson Orin AGX, to verify the
# image actually works on this hardware. CI's build-time guards can only
# check metadata (no GPU on the runner); these are the real-hardware checks
# from `runbooks/hephaestus-deploy.md` §2, packaged so they're a single
# command for end users.
#
# Usage (on Hephaestus dev daemon):
#   docker --host unix:///run/docker-dev.sock run --rm --runtime nvidia --gpus all \
#     ghcr.io/infernode-os/serving-sglang:orin-latest \
#     /opt/sglang/validate-on-hardware.sh
#
# Usage (field-deployment Orin AGX, single docker daemon):
#   docker run --rm --runtime nvidia --gpus all \
#     ghcr.io/infernode-os/serving-sglang:orin-latest \
#     /opt/sglang/validate-on-hardware.sh

set -euo pipefail

VENV=/opt/venv

step() { printf '\n=== %s ===\n' "$1"; }
ok()   { printf '  ok: %s\n' "$1"; }
fail() { printf '  FAIL: %s\n' "$1" >&2; exit 1; }

step "1. CUDA available + correct device"
${VENV}/bin/python3 - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    sys.exit(f"FAIL: cuda not available; torch={torch.__version__}")
cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f"  torch={torch.__version__} cuda={torch.version.cuda} device='{name}' capability={cap}")
if cap[0] < 7:
    sys.exit(f"FAIL: compute capability {cap} too old for SGLang")
PY
ok "torch + cuda + device"

step "2. SGLang upstream smoke test"
${VENV}/bin/python3 /opt/sglang/test.py
ok "upstream smoke"

step "3. gpt_oss model arch importable (INFR-77 acceptance)"
${VENV}/bin/python3 -c "import sglang.srt.models.gpt_oss as m; print('  ', m.__file__)"
ok "gpt_oss arch import"

step "4. sgl_kernel loadable on this device's compute capability"
${VENV}/bin/python3 -c "import sgl_kernel; assert sgl_kernel.common_ops is not None; print('  sgl_kernel: OK')"
ok "sgl_kernel arch-specific ops"

step "5. sglang.launch_server entrypoint"
${VENV}/bin/python3 -c "from sglang.srt.entrypoints.http_server import launch_server; print('  launch_server: OK')"
ok "launch_server entrypoint"

step "6. Tokenizer bake (INFR-78)"
for fam in llama-3 llama-3.1; do
  for f in tokenizer.json tokenizer_config.json special_tokens_map.json; do
    [[ -f /opt/tokenizers/$fam/$f ]] || fail "missing /opt/tokenizers/$fam/$f"
  done
done
ok "tokenizer dirs present"

printf '\nAll on-hardware checks passed.\n'
