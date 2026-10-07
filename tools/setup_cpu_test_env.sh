#!/usr/bin/env bash
# Install a Linux CPU environment that can run the Playground v2.5 unit tests
# and the tiny-model sdxl_train_network.py smoke test.
#
# This does not download Playground v2.5 or SDXL weights. The smoke test builds
# a randomly initialized SDXL-shaped checkpoint. CLIP tokenizers are small and
# are cached under ./tokenizer_cache for offline reruns.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

if ! "$PYTHON" -c "import torch, torchvision" >/dev/null 2>&1; then
  "$PYTHON" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

"$PYTHON" -m pip install -r "$ROOT/requirements-test-cpu.txt"
"$PYTHON" -m pip install -e "$ROOT"

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
mkdir -p "$ROOT/tokenizer_cache"
"$PYTHON" - << 'PY'
import os
from huggingface_hub import snapshot_download

dest_root = os.path.abspath("tokenizer_cache")
repos = [
    "openai/clip-vit-large-patch14",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
]
allow = ["tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json", "tokenizer_config.json"]
for repo in repos:
    local = os.path.join(dest_root, repo.replace("/", "_"))
    os.makedirs(local, exist_ok=True)
    snapshot_download(repo, local_dir=local, allow_patterns=allow)
    print("tokenizer cached:", local)
PY

echo "CPU test environment is ready."
echo "Unit tests: python -m pytest tests/test_playground_v25.py -q"
echo "Smoke test: python -m pytest tests/test_playground_v25_smoke.py -q"
