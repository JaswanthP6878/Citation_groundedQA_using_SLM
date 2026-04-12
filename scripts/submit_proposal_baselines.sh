#!/bin/bash -l

set -euo pipefail

QUESTION_LIMIT="${1:-100}"
SPLIT="${2:-validation}"
STRATEGY="${3:-section}"
MAX_INPUT_TOKENS="${4:-2048}"
MAX_CONTEXT_TOKENS="${5:-1200}"
MAX_NEW_TOKENS="${6:-128}"
GENERATION_MODEL="${7:-microsoft/Phi-3.5-mini-instruct}"
HOLD_ON="${8:-}"

mkdir -p logs

submit_baseline() {
  local label="$1"
  local retrieval_method="$2"
  local retrieval_k="$3"
  local retrieval_depth="$4"
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
    baseline
    "$SPLIT"
    "$STRATEGY"
    "$retrieval_k"
    "$retrieval_depth"
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
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$label" \
    "${job_id%%.*}" \
    "$retrieval_method" \
    "$STRATEGY" \
    "$log_path"
}

printf 'baseline\tjob_id\tretrieval\tstrategy\tlog\n'

JOB_PARAMETRIC="$(submit_baseline P none 0 0 "$HOLD_ON" | tee /dev/stderr | tail -n1 | cut -f2)"
JOB_BM25="$(submit_baseline M bm25 5 5 "$JOB_PARAMETRIC" | tee /dev/stderr | tail -n1 | cut -f2)"
submit_baseline R random 5 5 "$JOB_BM25" | tee /dev/stderr >/dev/null
