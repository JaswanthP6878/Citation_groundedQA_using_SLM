#!/bin/bash -l

set -euo pipefail

DEFAULT_PROJECT_ROOT="/projectnb/cs505am/students/$USER/zukhriddin_nlp_research_project"
if [[ ! -d "$DEFAULT_PROJECT_ROOT" && -d "/projectnb/cs505am/students/$USER/nlp_research_project" ]]; then
  DEFAULT_PROJECT_ROOT="/projectnb/cs505am/students/$USER/nlp_research_project"
fi

PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
TARGET_DIR="${1:-$PROJECT_ROOT/.vendor/qwen3_transformers_4_51}"

module load miniconda
conda activate vmr

mkdir -p "$TARGET_DIR"
python -m pip install --upgrade --target "$TARGET_DIR" \
  "transformers==4.51.3" \
  "tokenizers==0.21.1"

python - <<PY
import importlib.util
import sys
from pathlib import Path

target = Path("$TARGET_DIR")
sys.path.insert(0, str(target))
import transformers
import tokenizers
print({"target_dir": str(target), "transformers": transformers.__version__, "tokenizers": tokenizers.__version__})
PY
