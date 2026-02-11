#!/usr/bin/env bash
set -euo pipefail

# Downloads the latest Close Report artifact into ./reports/downloaded
# Requires: GitHub CLI (gh) and an authenticated session (gh auth login)

REPO="${REPO:-Shazil10/alpaca-paper-trader}"
WORKFLOW="${WORKFLOW:-close_report.yml}"
ARTIFACT_NAME="${ARTIFACT_NAME:-trading-reports-close}"
DEST_DIR="${DEST_DIR:-reports/downloaded}"

mkdir -p "$DEST_DIR"

if command -v gh >/dev/null 2>&1; then
  GH_BIN="gh"
elif [[ -x "/opt/homebrew/bin/gh" ]]; then
  GH_BIN="/opt/homebrew/bin/gh"
else
  echo "ERROR: gh not found on PATH, and /opt/homebrew/bin/gh does not exist." >&2
  echo "Install GitHub CLI: https://cli.github.com/" >&2
  exit 2
fi

# Prefer GitHub API queries via gh api (more robust than piping JSON into Python).
# Note: we intentionally don't rely on any external tools beyond gh itself.
#
# Extract owner/repo from $REPO (format: owner/name)
OWNER="${REPO%%/*}"
NAME="${REPO##*/}"

if [[ -z "$OWNER" || -z "$NAME" || "$OWNER" == "$NAME" ]]; then
  echo "ERROR: REPO must be in the form owner/name. Got: '$REPO'" >&2
  exit 2
fi

# Find the latest successful run for this workflow file.
RUN_ID=$(
  $GH_BIN api \
    -H "Accept: application/vnd.github+json" \
    "/repos/$OWNER/$NAME/actions/workflows/$WORKFLOW/runs?per_page=20" \
    --jq '.workflow_runs
      | map(select(.status=="completed" and .conclusion=="success"))
      | (.[0].id // empty)'
)

if [[ -z "${RUN_ID}" ]]; then
  echo "ERROR: No successful runs found for workflow '$WORKFLOW' in '$REPO'." >&2
  echo "Tip: confirm you're logged in: $GH_BIN auth status" >&2
  exit 3
fi

# Confirm the artifact exists on that run (helps catch caching / wrong workflow).
ARTIFACT_ID=$(
  $GH_BIN api \
    -H "Accept: application/vnd.github+json" \
    "/repos/$OWNER/$NAME/actions/runs/$RUN_ID/artifacts" \
    --jq ".artifacts | map(select(.name==\"$ARTIFACT_NAME\")) | (.[0].id // empty)"
)

if [[ -z "${ARTIFACT_ID}" ]]; then
  echo "ERROR: Artifact '$ARTIFACT_NAME' not found on run $RUN_ID." >&2
  echo "Tip: check the run page for uploaded artifacts." >&2
  exit 4
fi

# Clean destination and download.
rm -rf "$DEST_DIR"/*

$GH_BIN run download "$RUN_ID" \
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
