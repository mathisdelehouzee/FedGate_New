#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[error] tmux not found in PATH"
  exit 1
fi

SESSION_NAME="${SESSION_NAME:-fedgate_full_master}"
FORCE_RESTART="${FORCE_RESTART:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  if [ "${FORCE_RESTART}" = "1" ]; then
    tmux kill-session -t "${SESSION_NAME}"
  else
    echo "[error] session already exists: ${SESSION_NAME}"
    echo "Use FORCE_RESTART=1 to replace it."
    exit 1
  fi
fi

CMD="cd $(printf '%q' "${ROOT_DIR}") && RUN_ID=$(printf '%q' "${RUN_ID}") PREPARE_SPLITS=$(printf '%q' "${PREPARE_SPLITS:-1}") METHODS_CSV=$(printf '%q' "${METHODS_CSV:-centralized,fedavg,fedgate}") bash fedgate_full/run_all.sh"

tmux new-session -d -s "${SESSION_NAME}" "${CMD}"

echo "[ok] tmux session: ${SESSION_NAME}"
echo "[ok] run_id=${RUN_ID}"
echo "[ok] logs: ${ROOT_DIR}/fedgate_full/logs"
echo "Attach: tmux attach -t ${SESSION_NAME}"
