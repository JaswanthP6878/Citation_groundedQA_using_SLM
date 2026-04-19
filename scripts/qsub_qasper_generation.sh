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
DENSE_DEVICE_ARG="${12:-}"
RERANKER_DEVICE_ARG="${13:-}"
CITATION_VERIFY_ARG="${14:-}"
CITATION_VERIFIER_MODEL_ARG="${15:-}"
CITATION_VERIFIER_DEVICE_ARG="${16:-}"
CITATION_VERIFIER_THRESHOLD_ARG="${17:-}"

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
DENSE_DEVICE="${DENSE_DEVICE_ARG:-${DENSE_DEVICE:-}}"
RERANKER_DEVICE="${RERANKER_DEVICE_ARG:-${RERANKER_DEVICE:-}}"
CITATION_VERIFY="${CITATION_VERIFY_ARG:-${CITATION_VERIFY:-}}"
CITATION_VERIFIER_MODEL="${CITATION_VERIFIER_MODEL_ARG:-${CITATION_VERIFIER_MODEL:-cross-encoder/nli-deberta-v3-base}}"
CITATION_VERIFIER_DEVICE="${CITATION_VERIFIER_DEVICE_ARG:-${CITATION_VERIFIER_DEVICE:-}}"
CITATION_VERIFIER_THRESHOLD="${CITATION_VERIFIER_THRESHOLD_ARG:-${CITATION_VERIFIER_THRESHOLD:-0.5}}"
TRANSFORMERS_OVERLAY_DIR="${TRANSFORMERS_OVERLAY_DIR:-$PROJECT_ROOT/.vendor/qwen3_transformers_4_51}"
mkdir -p "$PROJECT_ROOT/logs" "$HF_HOME"

module load miniconda
conda activate vmr

cd "$PROJECT_ROOT"
export HF_HOME
export OMP_NUM_THREADS="${NSLOTS:-1}"

if [[ "$GENERATION_MODEL" == Qwen/Qwen3-* && -d "$TRANSFORMERS_OVERLAY_DIR" ]]; then
  export PYTHONPATH="$TRANSFORMERS_OVERLAY_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

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

if [[ -n "$DENSE_DEVICE" ]]; then
  cmd+=(--dense-device "$DENSE_DEVICE")
fi

if [[ -n "$RERANKER_DEVICE" ]]; then
  cmd+=(--reranker-device "$RERANKER_DEVICE")
fi

if [[ "$CITATION_VERIFY" == "1" || "$CITATION_VERIFY" == "true" || "$CITATION_VERIFY" == "yes" ]]; then
  cmd+=(--citation-verify)
  cmd+=(--citation-verifier-model "$CITATION_VERIFIER_MODEL")
  cmd+=(--citation-verifier-threshold "$CITATION_VERIFIER_THRESHOLD")
  if [[ -n "$CITATION_VERIFIER_DEVICE" ]]; then
    cmd+=(--citation-verifier-device "$CITATION_VERIFIER_DEVICE")
  fi
fi

"${cmd[@]}"
