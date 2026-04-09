#!/bin/bash -l

#$ -cwd
#$ -j y
#$ -P cs505am
#$ -q academic-gpu
#$ -l h_rt=02:00:00
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=6.0

set -euo pipefail

RETRIEVAL_METHOD="${1:-hybrid}"
PROMPT_STYLE="${2:-citation_forcing}"
SPLIT="${3:-validation}"
STRATEGY="${4:-section}"
RETRIEVAL_K="${5:-5}"
RETRIEVAL_DEPTH="${6:-20}"
QUESTION_LIMIT_ARG="${7:-}"
MAX_INPUT_TOKENS_ARG="${8:-}"
MAX_CONTEXT_TOKENS_ARG="${9:-}"
MAX_NEW_TOKENS_ARG="${10:-}"
GENERATION_MODEL_ARG="${11:-}"

DEFAULT_PROJECT_ROOT="/projectnb/cs505am/students/$USER/zukhriddin_nlp_research_project"
if [[ ! -d "$DEFAULT_PROJECT_ROOT" && -d "/projectnb/cs505am/students/$USER/nlp_research_project" ]]; then
  DEFAULT_PROJECT_ROOT="/projectnb/cs505am/students/$USER/nlp_research_project"
fi

PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
HF_HOME="${HF_HOME:-/projectnb/cs505am/students/$USER/.cache/huggingface}"
GENERATION_MODEL="${GENERATION_MODEL_ARG:-${GENERATION_MODEL:-microsoft/Phi-3.5-mini-instruct}}"
QUESTION_LIMIT="${QUESTION_LIMIT_ARG:-${QUESTION_LIMIT:-}}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS_ARG:-${MAX_INPUT_TOKENS:-2048}}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS_ARG:-${MAX_CONTEXT_TOKENS:-1200}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS_ARG:-${MAX_NEW_TOKENS:-128}}"
mkdir -p "$PROJECT_ROOT/logs" "$HF_HOME"

module load miniconda
conda activate vmr

cd "$PROJECT_ROOT"
export HF_HOME
export OMP_NUM_THREADS="${NSLOTS:-1}"

cmd=(
  python
  scripts/evaluate_qasper_generation.py
  --split "$SPLIT"
  --strategy "$STRATEGY"
  --retrieval-method "$RETRIEVAL_METHOD"
  --prompt-style "$PROMPT_STYLE"
  --retrieval-k "$RETRIEVAL_K"
  --retrieval-depth "$RETRIEVAL_DEPTH"
  --generation-model "$GENERATION_MODEL"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --max-input-tokens "$MAX_INPUT_TOKENS"
  --max-context-tokens "$MAX_CONTEXT_TOKENS"
)

if [[ -n "$QUESTION_LIMIT" ]]; then
  cmd+=(--question-limit "$QUESTION_LIMIT")
fi

"${cmd[@]}"
