#!/bin/bash -l

set -euo pipefail

QUESTION_LIMIT="${1:-100}"
SPLIT="${2:-validation}"
RETRIEVAL_K="${3:-5}"
RETRIEVAL_DEPTH="${4:-20}"
MAX_INPUT_TOKENS="${5:-2048}"
MAX_CONTEXT_TOKENS="${6:-1200}"
MAX_NEW_TOKENS="${7:-128}"
GENERATION_MODEL="${8:-microsoft/Phi-3.5-mini-instruct}"

mkdir -p logs

submit_condition() {
  local label="$1"
  local retrieval_method="$2"
  local prompt_style="$3"
  local strategy="$4"
  local hold_jid="${5:-}"
  local job_name="phi${label}${QUESTION_LIMIT}"
  local log_path="logs/${job_name}.log"
  local -a qsub_cmd=(
    qsub
    -terse
    -N "$job_name"
    -P cs505am
    -q academic-gpu
    -pe omp 1
    -l "h_rt=01:00:00,gpus=1,gpu_c=6.0"
    -o "$log_path"
  )

  if [[ -n "$hold_jid" ]]; then
    qsub_cmd+=(-hold_jid "$hold_jid")
  fi

  qsub_cmd+=(
    scripts/qsub_qasper_generation.sh
    "$retrieval_method"
    "$prompt_style"
    "$SPLIT"
    "$strategy"
    "$RETRIEVAL_K"
    "$RETRIEVAL_DEPTH"
    "$QUESTION_LIMIT"
    "$MAX_INPUT_TOKENS"
    "$MAX_CONTEXT_TOKENS"
    "$MAX_NEW_TOKENS"
    "$GENERATION_MODEL"
    cpu
    cpu
  )

  local job_id
  job_id="$("${qsub_cmd[@]}")"
  qalter -R y "${job_id%%.*}" >/dev/null 2>&1 || true
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" \
    "${job_id%%.*}" \
    "$retrieval_method" \
    "$prompt_style" \
    "$strategy" \
    "$log_path"
}

printf 'cond\tjob_id\tretrieval\tprompt\tstrategy\tlog\n'

JOB_A="$(submit_condition A dense baseline fixed | tee /dev/stderr | tail -n1 | cut -f2)"
JOB_B="$(submit_condition B hybrid baseline fixed "$JOB_A" | tee /dev/stderr | tail -n1 | cut -f2)"
JOB_C="$(submit_condition C hybrid baseline section "$JOB_B" | tee /dev/stderr | tail -n1 | cut -f2)"
JOB_D="$(submit_condition D hybrid_rerank baseline section "$JOB_C" | tee /dev/stderr | tail -n1 | cut -f2)"
submit_condition E hybrid_rerank citation_forcing section "$JOB_D" | tee /dev/stderr >/dev/null
