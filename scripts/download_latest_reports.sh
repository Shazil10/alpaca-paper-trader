#!/usr/bin/env bash
set -euo pipefail

# Downloads the latest Close Report artifact into ./reports/downloaded
# Requires: GitHub CLI (gh) and an authenticated session (gh auth login)

REPO="${REPO:-Shazil10/alpaca-paper-trader}"
WORKFLOW="${WORKFLOW:-close_report.yml}"
ARTIFACT_NAME="${ARTIFACT_NAME:-trading-reports-close}"
DEST_DIR="${DEST_DIR:-reports/downloaded}"

mkdir -p "$DEST_DIR"

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: gh not found. Install GitHub CLI first: https://cli.github.com/" >&2
  exit 2
fi

# Find latest successful run of the close report workflow.
RUN_ID=$(gh run list \
  --repo "$REPO" \
  --workflow "$WORKFLOW" \
  --json databaseId,conclusion,status \
  --limit 10 \
  | python - <<'PY'
import json,sys
runs=json.load(sys.stdin)
for r in runs:
    if r.get('conclusion')=='success' and r.get('status')=='completed':
        print(r['databaseId'])
        break
PY
)

if [[ -z "${RUN_ID}" ]]; then
  echo "ERROR: No successful runs found for $WORKFLOW in $REPO" >&2
  exit 3
fi

# Clean destination and download.
rm -rf "$DEST_DIR"/*

gh run download "$RUN_ID" \
  --repo "$REPO" \
  --name "$ARTIFACT_NAME" \
  --dir "$DEST_DIR"

echo "Downloaded artifact '$ARTIFACT_NAME' from run $RUN_ID into $DEST_DIR"

HTML_PATH="$DEST_DIR/orders_latest.html"
MD_PATH="$DEST_DIR/orders_latest.md"

if [[ -f "$HTML_PATH" ]]; then
  if command -v open >/dev/null 2>&1; then
    open "$HTML_PATH" >/dev/null 2>&1 || true
  fi
  echo "Opened: $HTML_PATH"
elif [[ -f "$MD_PATH" ]]; then
  echo "Report downloaded (HTML missing). Open in VS Code: $MD_PATH"
else
  echo "Report downloaded, but expected files not found in $DEST_DIR" >&2
fi
