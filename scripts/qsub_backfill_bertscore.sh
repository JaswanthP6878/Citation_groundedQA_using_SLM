#!/bin/bash -l

#$ -cwd
#$ -j y
#$ -P cs505am
#$ -q academic-gpu
#$ -l h_rt=01:00:00
#$ -pe omp 1
#$ -l gpus=1
#$ -l gpu_c=6.0

set -euo pipefail

RUN_PATTERN="${1:-generation/**/*.json}"
MODEL_TYPE="${2:-roberta-large}"
BATCH_SIZE="${3:-16}"

DEFAULT_PROJECT_ROOT="/projectnb/cs505am/students/$USER/zukhriddin_nlp_research_project"
if [[ ! -d "$DEFAULT_PROJECT_ROOT" && -d "/projectnb/cs505am/students/$USER/nlp_research_project" ]]; then
  DEFAULT_PROJECT_ROOT="/projectnb/cs505am/students/$USER/nlp_research_project"
fi

PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
HF_HOME="${HF_HOME:-/projectnb/cs505am/students/$USER/.cache/huggingface}"
BERTSCORE_OVERLAY_DIR="${BERTSCORE_OVERLAY_DIR:-$PROJECT_ROOT/.vendor/bertscore}"
mkdir -p "$PROJECT_ROOT/logs" "$HF_HOME"

module load miniconda
conda activate vmr

cd "$PROJECT_ROOT"
export HF_HOME
export OMP_NUM_THREADS="${NSLOTS:-1}"

if [[ -d "$BERTSCORE_OVERLAY_DIR" ]]; then
  export PYTHONPATH="$BERTSCORE_OVERLAY_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

python scripts/backfill_bertscore.py \
  --run-root "$PROJECT_ROOT/runs" \
  --pattern "$RUN_PATTERN" \
  --model-type "$MODEL_TYPE" \
  --device cuda \
  --batch-size "$BATCH_SIZE" \
  --skip-existing
