#!/usr/bin/env bash
#
# Bake a curated set of non-gated tokenizer + chat-template directories
# into the SGLang container at /opt/tokenizers/<family>/.
#
# Why: SGLang's GGUF tokenizer doesn't register Llama-3's special tokens
# (<|eot_id|>, <|begin_of_text|>, <|start_header_id|>) properly, so stops
# don't match cleanly and chat-template framing is wrong. The fix is to
# point --tokenizer-path at a real HuggingFace tokenizer dir at launch
# time (see runbooks/hephaestus-deploy.md). Baking the tokenizers into
# the image means the launch command is fully offline-capable.
#
# Each family pulled is the tokenizer files only (~30 MB per family);
# weights are NOT downloaded. We pick non-gated mirrors so no HF login
# is required at build time.
#
# Owning ticket: INFR-78.

set -euo pipefail

DEST="${TOKENIZER_DIR:-/opt/tokenizers}"
mkdir -p "$DEST"

# family       repo (non-gated mirror)                     alias dir under $DEST
declare -A FAMILIES=(
  [llama-3.1]="unsloth/Meta-Llama-3.1-8B-Instruct"
  [llama-3]="NousResearch/Meta-Llama-3-8B-Instruct"
)

# Tokenizer-only files. No weights, no model.safetensors.
PATTERNS=(
  "tokenizer.json"
  "tokenizer_config.json"
  "special_tokens_map.json"
  "chat_template.json"
  "generation_config.json"
)

python3 - "$DEST" "${!FAMILIES[@]}" <<'PY' "${FAMILIES[@]}"
import os, sys
from huggingface_hub import snapshot_download

dest_root = sys.argv[1]
families = sys.argv[2:]
# Args come in two halves: aliases then repos (same order).
n = len(families) // 2
aliases, repos = families[:n], families[n:]

patterns = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.json",
    "generation_config.json",
]

for alias, repo in zip(aliases, repos):
    target = os.path.join(dest_root, alias)
    os.makedirs(target, exist_ok=True)
    print(f"[bake-tokenizers] {repo} -> {target}")
    snapshot_download(
        repo_id=repo,
        local_dir=target,
        local_dir_use_symlinks=False,
        allow_patterns=patterns,
    )

print("[bake-tokenizers] done")
PY

# Strip HuggingFace cache metadata; we only want the flat tokenizer dirs.
find "$DEST" -name '.huggingface' -prune -exec rm -rf {} + 2>/dev/null || true
du -sh "$DEST"/* 2>/dev/null || true
