#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_ROOT="${1:-$REPO_ROOT/local_scc_results}"
SCC_USER="${SCC_USER:-zfmsai}"
SCC_HOST="${SCC_HOST:-scc1.bu.edu}"
SCC_STUDENT_ROOT="${SCC_STUDENT_ROOT:-/projectnb/cs505am/students/$SCC_USER}"

REMOTE_PROJECT_ROOT=""
for candidate in \
  "$SCC_STUDENT_ROOT/zukhriddin_nlp_research_project" \
  "$SCC_STUDENT_ROOT/nlp_research_project"
do
  if ssh "$SCC_USER@$SCC_HOST" "test -d '$candidate'"; then
    REMOTE_PROJECT_ROOT="$candidate"
    break
  fi
done

if [[ -z "$REMOTE_PROJECT_ROOT" ]]; then
  echo "Could not find SCC project root under $SCC_STUDENT_ROOT" >&2
  exit 1
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SNAPSHOT_ROOT="$LOCAL_ROOT/snapshots/$TIMESTAMP"
LATEST_ROOT="$LOCAL_ROOT/latest"

mkdir -p "$SNAPSHOT_ROOT" "$LATEST_ROOT"

sync_subtree() {
  local remote_subpath="$1"
  local local_target="$2"
  mkdir -p "$local_target"
  rsync -az \
    "$SCC_USER@$SCC_HOST:$REMOTE_PROJECT_ROOT/$remote_subpath/" \
    "$local_target/" || true
}

sync_subtree "runs" "$SNAPSHOT_ROOT/runs"
sync_subtree "logs" "$SNAPSHOT_ROOT/logs"
sync_subtree "runs" "$LATEST_ROOT/runs"
sync_subtree "logs" "$LATEST_ROOT/logs"

cat > "$SNAPSHOT_ROOT/manifest.txt" <<EOF
timestamp_utc=$TIMESTAMP
remote_project_root=$REMOTE_PROJECT_ROOT
scc_host=$SCC_HOST
scc_user=$SCC_USER
EOF

cp "$SNAPSHOT_ROOT/manifest.txt" "$LATEST_ROOT/manifest.txt"

echo "Archived SCC results to:"
echo "  $SNAPSHOT_ROOT"
echo "Updated latest mirror:"
echo "  $LATEST_ROOT"
