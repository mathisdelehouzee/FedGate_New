#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <log_path> <diagnostics_path> -- <command...>" >&2
  exit 2
fi

LOG_PATH="$1"
DIAG_PATH="$2"
shift 2

if [ "$1" != "--" ]; then
  echo "Expected -- before command" >&2
  exit 2
fi
shift

if [ "$#" -eq 0 ]; then
  echo "Missing command" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG_PATH")" "$(dirname "$DIAG_PATH")"

export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1

{
  echo "[run_with_diagnostics] start=$(date --iso-8601=seconds)"
  echo "[run_with_diagnostics] cwd=$(pwd)"
  echo "[run_with_diagnostics] cmd=$*"
} | tee "$LOG_PATH"

"$@" 2>&1 | tee -a "$LOG_PATH"
CMD_STATUS=${PIPESTATUS[0]}

if [ "$CMD_STATUS" -ne 0 ]; then
  {
    echo "[diagnostics] crash_time=$(date --iso-8601=seconds)"
    echo "[diagnostics] exit_code=$CMD_STATUS"
    echo
    echo "[diagnostics] free -h"
    free -h || true
    echo
    echo "[diagnostics] /proc/meminfo"
    grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree|Shmem' /proc/meminfo || true
    echo
    echo "[diagnostics] nvidia-smi"
    nvidia-smi || true
    echo
    echo "[diagnostics] python processes"
    ps -eo pid,ppid,etime,pcpu,pmem,rss,vsz,cmd | rg 'python|run_fedavg|run_fedgate|run_centralized_baseline|run_federated_benchmark' || true
    echo
    echo "[diagnostics] result files"
    find results -maxdepth 4 -type f | sort | tail -n 200 || true
  } | tee "$DIAG_PATH" | tee -a "$LOG_PATH"
fi

exit "$CMD_STATUS"
