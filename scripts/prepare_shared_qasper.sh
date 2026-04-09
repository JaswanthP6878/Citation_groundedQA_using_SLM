#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT="${1:-/projectnb/cs505am/projects/nlp_research_project/qasper}"
RAW_DIR="${TARGET_ROOT}/raw"
EXTRACTED_DIR="${TARGET_ROOT}/extracted"
PROCESSED_DIR="${TARGET_ROOT}/processed"
TRAIN_DEV_URL="https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
TEST_URL="https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"

umask 002

mkdir -p "${RAW_DIR}" "${EXTRACTED_DIR}" "${PROCESSED_DIR}"
chmod 2775 "${TARGET_ROOT}" "${RAW_DIR}" "${EXTRACTED_DIR}" "${PROCESSED_DIR}"

download_if_missing() {
  local url="$1"
  local destination="$2"
  if [[ -f "${destination}" ]]; then
    echo "Already present: ${destination}"
    return 0
  fi

  echo "Downloading ${url}"
  wget -O "${destination}.part" "${url}"
  mv "${destination}.part" "${destination}"
}

download_if_missing "${TRAIN_DEV_URL}" "${RAW_DIR}/qasper-train-dev-v0.3.tgz"
download_if_missing "${TEST_URL}" "${RAW_DIR}/qasper-test-and-evaluator-v0.3.tgz"

if [[ ! -f "${EXTRACTED_DIR}/qasper-train-v0.3.json" ]]; then
  tar -xzf "${RAW_DIR}/qasper-train-dev-v0.3.tgz" -C "${EXTRACTED_DIR}"
fi

if [[ ! -f "${EXTRACTED_DIR}/qasper-test-v0.3.json" ]]; then
  tar -xzf "${RAW_DIR}/qasper-test-and-evaluator-v0.3.tgz" -C "${EXTRACTED_DIR}"
fi

find "${TARGET_ROOT}" -type d -exec chmod 2775 {} +
find "${TARGET_ROOT}" -type f -exec chmod 664 {} +

cat <<EOF
Shared QASPER dataset prepared at:
  ${TARGET_ROOT}

Raw archives:
  ${RAW_DIR}

Extracted JSON files:
  ${EXTRACTED_DIR}/qasper-train-v0.3.json
  ${EXTRACTED_DIR}/qasper-dev-v0.3.json
  ${EXTRACTED_DIR}/qasper-test-v0.3.json

Processed artifact root:
  ${PROCESSED_DIR}
EOF
