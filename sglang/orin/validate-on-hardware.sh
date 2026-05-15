#!/usr/bin/env bash
# On-hardware validator for the InferNode SGLang image. Run after pulling
# the published image to a Jetson Orin AGX. CI's build-time guards can only
# check metadata (no GPU on the runner); these are the real-hardware checks.
#
# Includes a real serving smoke that launches sglang.launch_server with a
# small model, asserts /health, asserts the "KV Cache is allocated" log
# line, and exercises /v1/chat/completions. This is the gate that catches
# untested-upstream regressions before they reach a deploy — the kind of
# breakage that previously slipped past the import-only checks.
#
# Usage (on Hephaestus dev daemon, with an HF cache mount):
#   docker --host unix:///run/docker-dev.sock run --rm \
#     --runtime nvidia --gpus all \
#     -v /mnt/orin-ssd/huggingface:/root/.cache/huggingface \
#     -e HF_HOME=/root/.cache/huggingface \
#     ghcr.io/infernode-os/serving-sglang:orin-latest \
#     /opt/sglang/validate-on-hardware.sh
#
# Usage (field-deployment Orin AGX, single docker daemon):
#   docker run --rm --runtime nvidia --gpus all \
#     -v /var/lib/huggingface:/root/.cache/huggingface \
#     -e HF_HOME=/root/.cache/huggingface \
#     ghcr.io/infernode-os/serving-sglang:orin-latest \
#     /opt/sglang/validate-on-hardware.sh
#
# The HF mount is recommended (re-uses ~2 GB TinyLlama between runs). If
# absent, the smoke will pull TinyLlama fresh into the container each time.

set -euo pipefail

SMOKE_MODEL="${SMOKE_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
SMOKE_PORT="${SMOKE_PORT:-30099}"
SMOKE_HOST="127.0.0.1"
SMOKE_LOG="/tmp/sglang-smoke.log"
SMOKE_TIMEOUT_SEC="${SMOKE_TIMEOUT_SEC:-180}"

step() { printf '\n=== %s ===\n' "$1"; }
ok()   { printf '  ok: %s\n' "$1"; }
fail() { printf '  FAIL: %s\n' "$1" >&2; exit 1; }

step "1. CUDA available + correct device"
python3 - <<'PY'
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
python3 /opt/sglang/test.py
ok "upstream smoke"

step "3. gpt_oss model arch importable (informational)"
# v1 ships SGLang 0.4.1.post7 which has no gpt_oss.py; INFR-92 tracks the
# upgrade. Treat as informational, not a failure.
if python3 -c "import sglang.srt.models.gpt_oss as m; print('  ', m.__file__)" 2>/dev/null; then
    ok "gpt_oss arch present (INFR-92 upgrade landed)"
else
    printf '  skip: gpt_oss not present in this SGLang version (expected for v1; INFR-92 covers the upgrade)\n'
fi

step "4. sgl_kernel loadable on this device's compute capability"
# Import succeeds only if the wheel ships kernels for this device's arch
# (sm_87 on Orin). A wrong-arch wheel fails here at the .so load. We probe
# whichever ops attribute the version exposes — older sgl_kernel uses
# `ops`, newer (0.5+) uses `common_ops`.
python3 -c "
import sgl_kernel, importlib.metadata as m
v = m.version('sgl-kernel')
ops = getattr(sgl_kernel, 'ops', None) or getattr(sgl_kernel, 'common_ops', None)
assert ops is not None, f'sgl_kernel {v} has neither .ops nor .common_ops'
print(f'  sgl_kernel: {v} (ops module loaded)')
"
ok "sgl_kernel arch-specific ops"

step "5. sglang.launch_server entrypoint importable"
python3 -c "from sglang.srt.server import launch_server; print('  launch_server: OK')" 2>/dev/null \
    || python3 -c "from sglang.srt.entrypoints.http_server import launch_server; print('  launch_server: OK')"
ok "launch_server entrypoint"

step "6. Tokenizer bake (INFR-78)"
for fam in llama-3 llama-3.1; do
    for f in tokenizer.json tokenizer_config.json special_tokens_map.json; do
        [[ -f /opt/tokenizers/$fam/$f ]] || fail "missing /opt/tokenizers/$fam/$f"
    done
done
ok "tokenizer dirs present"

step "7. Serving smoke (TinyLlama → /v1/chat/completions, KV-cache verified)"
# Launch sglang.launch_server in the background and exercise the actual
# HTTP path. Verifies: server starts, KV cache is allocated, /health 200s,
# /v1/chat/completions returns a sensible completion.
#
# Flags mirror runbooks/hephaestus-deploy.md §3 — keeps the smoke aligned
# with the documented production launch shape.

rm -f "$SMOKE_LOG"
python3 -m sglang.launch_server \
    --model-path "$SMOKE_MODEL" \
    --host "$SMOKE_HOST" --port "$SMOKE_PORT" \
    --attention-backend triton \
    --mem-fraction-static 0.5 \
    --disable-cuda-graph \
    --log-level info \
    > "$SMOKE_LOG" 2>&1 &
SGLANG_PID=$!

# Make sure we never leak the background server, even on early fail.
cleanup() {
    if kill -0 "$SGLANG_PID" 2>/dev/null; then
        kill -TERM "$SGLANG_PID" 2>/dev/null || true
        for _ in $(seq 1 15); do
            kill -0 "$SGLANG_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$SGLANG_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

printf '  waiting up to %ss for /health (pid=%s, log=%s)\n' "$SMOKE_TIMEOUT_SEC" "$SGLANG_PID" "$SMOKE_LOG"

# Poll /health. Also tail the log so a serve failure surfaces quickly.
READY=0
for i in $(seq 1 "$SMOKE_TIMEOUT_SEC"); do
    if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
        printf '  launch_server exited; last 80 lines of log:\n'
        tail -n 80 "$SMOKE_LOG" >&2 || true
        fail "launch_server died before /health came up"
    fi
    if python3 -c "
import urllib.request, sys
try:
    with urllib.request.urlopen('http://${SMOKE_HOST}:${SMOKE_PORT}/health', timeout=2) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(2)
" 2>/dev/null; then
        READY=1
        printf '  /health up after %ss\n' "$i"
        break
    fi
    sleep 1
done
[[ "$READY" == "1" ]] || { tail -n 80 "$SMOKE_LOG" >&2 || true; fail "/health did not come up within ${SMOKE_TIMEOUT_SEC}s"; }

# KV cache assertion (user-requested gate, per docs/SGLANG-ADOPTION-NOTES.md
# launch-time log line). If this line is missing, the runtime started but
# didn't allocate a KV cache — a silent regression we'd rather not ship.
if grep -q "KV Cache is allocated" "$SMOKE_LOG"; then
    KV_LINE=$(grep "KV Cache is allocated" "$SMOKE_LOG" | head -1)
    printf '  KV cache log line: %s\n' "$KV_LINE"
    ok "KV cache allocated"
else
    tail -n 80 "$SMOKE_LOG" >&2 || true
    fail "no 'KV Cache is allocated' line in launch log"
fi

# /v1/chat/completions round-trip
printf '  POST /v1/chat/completions\n'
RESP=$(python3 - <<PY
import json, urllib.request
req = urllib.request.Request(
    "http://${SMOKE_HOST}:${SMOKE_PORT}/v1/chat/completions",
    data=json.dumps({
        "model": "${SMOKE_MODEL}",
        "messages": [{"role": "user", "content": "2+2="}],
        "max_tokens": 16,
        "temperature": 0,
    }).encode(),
    headers={"content-type": "application/json"},
)
with urllib.request.urlopen(req, timeout=60) as r:
    body = json.loads(r.read())
content = body["choices"][0]["message"]["content"]
usage = body.get("usage", {})
print(f"  completion: {content!r}")
print(f"  usage: prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}")
assert content.strip(), "empty completion"
PY
)
printf '%s\n' "$RESP"
ok "/v1/chat/completions returned non-empty completion"

# Graceful shutdown — trap will hard-kill if it stalls.
kill -TERM "$SGLANG_PID" 2>/dev/null || true
wait "$SGLANG_PID" 2>/dev/null || true
trap - EXIT

printf '\nAll on-hardware checks passed.\n'
