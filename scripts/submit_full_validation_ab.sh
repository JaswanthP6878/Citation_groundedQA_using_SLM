#!/bin/bash -l

set -euo pipefail

SPLIT="${1:-validation}"
RETRIEVAL_K="${2:-5}"
RETRIEVAL_DEPTH="${3:-20}"
MAX_INPUT_TOKENS="${4:-2048}"
MAX_CONTEXT_TOKENS="${5:-1200}"
MAX_NEW_TOKENS="${6:-128}"
GENERATION_MODEL="${7:-microsoft/Phi-3.5-mini-instruct}"
WALLTIME="${8:-06:00:00}"

mkdir -p logs

submit_job() {
  local label="$1"
  local retrieval_method="$2"
  local strategy="$3"
  local hold_jid="${4:-}"
  local job_name="phiFull${label}"
  local log_path="logs/${job_name}.log"
  local -a qsub_cmd=(
    qsub
    -terse
    -N "$job_name"
    -P cs505am
    -q academic-gpu
    -pe omp 1
    -l "h_rt=${WALLTIME},gpus=1,gpu_c=6.0"
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
    "$strategy"
    "$RETRIEVAL_K"
    "$RETRIEVAL_DEPTH"
    ""
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
    "$strategy" \
    "$log_path"
}

printf 'label\tjob_id\tretrieval\tstrategy\tlog\n'

JOB_A="$(submit_job A dense fixed | tee /dev/stderr | tail -n1 | cut -f2)"
JOB_B="$(submit_job B hybrid fixed "$JOB_A" | tee /dev/stderr | tail -n1 | cut -f2)"

qsub -terse \
  -N bertfillAB \
  -P cs505am \
  -q academic-gpu \
  -pe omp 1 \
  -l h_rt=01:00:00,gpus=1,gpu_c=6.0 \
  -o logs/bertfillAB.log \
  -hold_jid "$JOB_B" \
  scripts/qsub_backfill_bertscore.sh \
  "generation/validation/fixed/*.json" \
  roberta-large \
  16 | tee /dev/stderr >/dev/null
