#!/bin/bash -l

set -euo pipefail

DEFAULT_PROJECT_ROOT="/projectnb/cs505am/students/$USER/zukhriddin_nlp_research_project"
if [[ ! -d "$DEFAULT_PROJECT_ROOT" && -d "/projectnb/cs505am/students/$USER/nlp_research_project" ]]; then
  DEFAULT_PROJECT_ROOT="/projectnb/cs505am/students/$USER/nlp_research_project"
fi

PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
TARGET_DIR="${1:-$PROJECT_ROOT/.vendor/bertscore}"

module load miniconda
conda activate vmr

mkdir -p "$TARGET_DIR"
python -m pip install --upgrade --target "$TARGET_DIR" --no-deps \
  "bert-score==0.3.13"
python -m pip install --upgrade --target "$TARGET_DIR" \
  matplotlib pandas packaging requests tqdm
rm -rf "$TARGET_DIR"/numpy "$TARGET_DIR"/numpy-*.dist-info

python - <<PY
import sys
from pathlib import Path
import numpy

target = Path("$TARGET_DIR")
sys.path.insert(0, str(target))
import bert_score
print({"target_dir": str(target), "numpy": numpy.__version__, "bert_score": bert_score.__version__})
PY
