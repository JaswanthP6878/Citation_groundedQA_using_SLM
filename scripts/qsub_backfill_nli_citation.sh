#!/bin/bash -l

#$ -cwd
#$ -j y
#$ -P cs505am
#$ -q academic-gpu
#$ -l h_rt=03:00:00
#$ -pe omp 1
#$ -l gpus=1
#$ -l gpu_c=6.0

set -euo pipefail

DEFAULT_PROJECT_ROOT="/projectnb/cs505am/students/$USER/zukhriddin_nlp_research_project"
if [[ ! -d "$DEFAULT_PROJECT_ROOT" && -d "/projectnb/cs505am/students/$USER/nlp_research_project" ]]; then
  DEFAULT_PROJECT_ROOT="/projectnb/cs505am/students/$USER/nlp_research_project"
fi

PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
HF_HOME="${HF_HOME:-/projectnb/cs505am/students/$USER/.cache/huggingface}"
NLI_MODEL_NAME="${NLI_MODEL_NAME:-cross-encoder/nli-deberta-v3-base}"
NLI_THRESHOLD="${NLI_THRESHOLD:-0.5}"
mkdir -p "$PROJECT_ROOT/logs" "$HF_HOME"

module load miniconda
conda activate vmr

cd "$PROJECT_ROOT"
export HF_HOME
export OMP_NUM_THREADS="${NSLOTS:-1}"

python scripts/backfill_nli_citation.py \
  --run-root "$PROJECT_ROOT/runs" \
  --processed-root "/projectnb/cs505am/projects/zukhriddin_nlp_research_project/qasper/processed" \
  --pattern "generation/**/*.json" \
  --model-name "$NLI_MODEL_NAME" \
  --threshold "$NLI_THRESHOLD" \
  --skip-existing
