#!/bin/bash -l

#$ -cwd
#$ -j y
#$ -P cs505am
#$ -l h_rt=02:00:00
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=6.0

set -euo pipefail

METHOD="${1:-dense}"
SPLIT="${2:-validation}"
STRATEGY="${3:-section}"
TOP_K="${4:-5}"
MRR_K="${5:-10}"
RETRIEVAL_DEPTH="${6:-20}"

PROJECT_ROOT="${PROJECT_ROOT:-/projectnb/cs505am/students/$USER/nlp_research_project}"
HF_HOME="${HF_HOME:-/projectnb/cs505am/students/$USER/.cache/huggingface}"
mkdir -p "$PROJECT_ROOT/logs" "$HF_HOME"

module load miniconda
conda activate vmr

cd "$PROJECT_ROOT"
export HF_HOME
export OMP_NUM_THREADS="${NSLOTS:-1}"

python scripts/evaluate_qasper_retrieval.py \
  --split "$SPLIT" \
  --strategy "$STRATEGY" \
  --method "$METHOD" \
  --top-k "$TOP_K" \
  --mrr-k "$MRR_K" \
  --retrieval-depth "$RETRIEVAL_DEPTH"
