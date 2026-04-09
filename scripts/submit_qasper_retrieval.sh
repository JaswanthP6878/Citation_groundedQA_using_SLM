#!/bin/bash -l

set -euo pipefail

METHOD="${1:-dense}"
SPLIT="${2:-validation}"
STRATEGY="${3:-section}"
TOP_K="${4:-5}"
MRR_K="${5:-10}"
RETRIEVAL_DEPTH="${6:-20}"

PROJECT_ROOT="${PROJECT_ROOT:-/projectnb/cs505am/students/$USER/nlp_research_project}"
QSUB_SCRIPT="${PROJECT_ROOT}/scripts/qsub_qasper_retrieval.sh"
LOG_DIR="${PROJECT_ROOT}/logs"
SCC_SUBMIT_HOST="${SCC_SUBMIT_HOST:-sccsvc.bu.edu}"

mkdir -p "$LOG_DIR"

if [[ ! -f "$QSUB_SCRIPT" ]]; then
  echo "missing batch script: $QSUB_SCRIPT" >&2
  exit 1
fi

JOB_NAME="${JOB_NAME:-qasper_${METHOD}_${SPLIT}_${STRATEGY}}"
JOB_NAME="${JOB_NAME//[^[:alnum:]_-]/_}"
LOG_PATH="${LOG_DIR}/${JOB_NAME}.log"

build_qsub_cmd() {
  local -a cmd=(
    qsub
    -N "$JOB_NAME"
    -o "$LOG_PATH"
    "$QSUB_SCRIPT"
    "$METHOD"
    "$SPLIT"
    "$STRATEGY"
    "$TOP_K"
    "$MRR_K"
    "$RETRIEVAL_DEPTH"
  )
  printf '%q ' "${cmd[@]}"
}

run_local_submit() {
  local output
  set +e
  output="$("$@" 2>&1)"
  local status=$?
  set -e
  printf '%s' "$output"
  return "$status"
}

LOCAL_CMD=(
  qsub
  -N "$JOB_NAME"
  -o "$LOG_PATH"
  "$QSUB_SCRIPT"
  "$METHOD"
  "$SPLIT"
  "$STRATEGY"
  "$TOP_K"
  "$MRR_K"
  "$RETRIEVAL_DEPTH"
)

LOCAL_OUTPUT="$(run_local_submit "${LOCAL_CMD[@]}")" || LOCAL_STATUS=$?
LOCAL_STATUS="${LOCAL_STATUS:-0}"

if [[ "$LOCAL_STATUS" -eq 0 ]]; then
  printf '%s\n' "$LOCAL_OUTPUT"
  exit 0
fi

if [[ "$LOCAL_OUTPUT" != *"no submit host"* ]]; then
  printf '%s\n' "$LOCAL_OUTPUT" >&2
  exit "$LOCAL_STATUS"
fi

printf '%s\n' "$LOCAL_OUTPUT" >&2
printf 'falling back to %s for qsub submission\n' "$SCC_SUBMIT_HOST" >&2

REMOTE_QSUB_CMD="$(build_qsub_cmd)"
printf -v REMOTE_SHELL 'cd %q && %s' "$PROJECT_ROOT" "$REMOTE_QSUB_CMD"

exec ssh -tt "${USER}@${SCC_SUBMIT_HOST}" "$REMOTE_SHELL"
