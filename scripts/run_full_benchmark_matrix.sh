#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="${PYTHON_BIN}"
elif [[ -x "${ROOT_DIR}/../FedGate/.venv/bin/python" ]]; then
  PYTHON="${ROOT_DIR}/../FedGate/.venv/bin/python"
else
  PYTHON="python3"
fi

RUN_WITH_DIAG="${ROOT_DIR}/scripts/run_with_diagnostics.sh"
GENERATE_SPLITS="${ROOT_DIR}/scripts/generate_scenario_splits.py"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/full_matrix_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FED_BATCH_SIZE="${FED_BATCH_SIZE:-0}"
CENTRAL_BATCH_SIZE="${CENTRAL_BATCH_SIZE:-0}"
RUN_CENTRALIZED="${RUN_CENTRALIZED:-1}"
RUN_FEDAVG="${RUN_FEDAVG:-1}"
RUN_FEDGATE="${RUN_FEDGATE:-1}"
LIMIT_ROWS="${LIMIT_ROWS:-0}"
SEEDS="${SEEDS:-}"
OVERWRITE_SPLITS="${OVERWRITE_SPLITS:-0}"

mkdir -p "${LOG_DIR}"

CENTRAL_CONFIG="configs/centralized_multimodal.yaml"
FED_CONFIGS=(
  "configs/s0_congruent_iid.yaml"
  "configs/s1_congruent_non_iid.yaml"
  "configs/s2_non_congruent_iid.yaml"
  "configs/s3_non_congruent_non_iid.yaml"
)

run_logged() {
  local label="$1"
  shift
  local log_path="${LOG_DIR}/${label}.log"
  local diag_path="${LOG_DIR}/${label}.diagnostics.log"
  echo "[run_full_benchmark_matrix] ${label}"
  "${RUN_WITH_DIAG}" "${log_path}" "${diag_path}" -- "$@"
}

append_common_args() {
  local -n _cmd_ref=$1
  _cmd_ref+=("--device" "${DEVICE}")
  _cmd_ref+=("--num-workers" "${NUM_WORKERS}")
  if [[ -n "${SEEDS}" ]]; then
    # shellcheck disable=SC2206
    local seeds_array=( ${SEEDS} )
    _cmd_ref+=("--seeds" "${seeds_array[@]}")
  fi
}

precompute_if_needed() {
  local config_path="$1"
  local info
  local cache_dir
  local row_count
  local existing_count
  info="$("${PYTHON}" - <<'PY' "${config_path}"
import csv
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1]).resolve()
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
data_cfg = dict(cfg.get("data", {}))
cache_cfg = dict(data_cfg.get("mri_cache", {}))
raw_cache_dir = cache_cfg.get("dir", "")
raw_csv = data_cfg.get("csv", "")
cache_dir = Path(raw_cache_dir).expanduser()
csv_path = Path(raw_csv).expanduser()
if not cache_dir.is_absolute():
    cache_dir = (config_path.parent / cache_dir).resolve()
if not csv_path.is_absolute():
    csv_path = (config_path.parent / csv_path).resolve()
with csv_path.open(encoding="utf-8") as handle:
    row_count = sum(1 for _ in csv.DictReader(handle))
print(f"{cache_dir}\t{row_count}")
PY
)"
  IFS=$'\t' read -r cache_dir row_count <<<"${info}"
  existing_count=0
  if [[ -d "${cache_dir}" ]]; then
    existing_count="$(find "${cache_dir}" -maxdepth 1 -type f | wc -l | tr -d ' ')"
  fi
  if [[ "${existing_count}" -lt "${row_count}" ]]; then
    local label="precompute_$(basename "${config_path}" .yaml)"
    local cmd=("${PYTHON}" -u "scripts/precompute_mri_cache.py" "--config" "${config_path}")
    if [[ "${LIMIT_ROWS}" != "0" ]]; then
      cmd+=("--limit-rows" "${LIMIT_ROWS}")
    fi
    run_logged "${label}" "${cmd[@]}"
  else
    echo "[run_full_benchmark_matrix] cache ok for ${config_path} (${existing_count}/${row_count})"
  fi
}

declare -A SEEN_CACHE=()

generate_splits_if_needed() {
  local config_path="$1"
  local info
  local manifest_path
  local report_path
  info="$("${PYTHON}" - <<'PY' "${config_path}"
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1]).resolve()
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
artifacts_root = Path(cfg["paths"]["artifacts_root"]).expanduser()
if not artifacts_root.is_absolute():
    artifacts_root = (config_path.parent / artifacts_root).resolve()
print(artifacts_root / "splits" / "splits_manifest.json")
print(artifacts_root / "splits" / "splits_report.json")
PY
)"
  manifest_path="$(printf '%s\n' "${info}" | sed -n '1p')"
  report_path="$(printf '%s\n' "${info}" | sed -n '2p')"
  if [[ "${OVERWRITE_SPLITS}" == "1" || ! -f "${manifest_path}" || ! -f "${report_path}" ]]; then
    local label="splits_$(basename "${config_path}" .yaml)"
    local cmd=("${PYTHON}" -u "${GENERATE_SPLITS}" "--config" "${config_path}")
    if [[ "${OVERWRITE_SPLITS}" == "1" ]]; then
      cmd+=("--overwrite")
    fi
    run_logged "${label}" "${cmd[@]}"
  else
    echo "[run_full_benchmark_matrix] splits ok for ${config_path}"
  fi
}

precompute_once_for_config() {
  local config_path="$1"
  local cache_dir
  cache_dir="$("${PYTHON}" - <<'PY' "${config_path}"
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1]).resolve()
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
cache_dir = Path(cfg["data"]["mri_cache"]["dir"]).expanduser()
if not cache_dir.is_absolute():
    cache_dir = (config_path.parent / cache_dir).resolve()
print(cache_dir)
PY
)"
  if [[ -z "${SEEN_CACHE["${cache_dir}"]:-}" ]]; then
    SEEN_CACHE["${cache_dir}"]=1
    precompute_if_needed "${config_path}"
  else
    echo "[run_full_benchmark_matrix] cache already handled for ${config_path}"
  fi
}

for config_path in "${FED_CONFIGS[@]}"; do
  generate_splits_if_needed "${config_path}"
done

precompute_once_for_config "${CENTRAL_CONFIG}"
for config_path in "${FED_CONFIGS[@]}"; do
  precompute_once_for_config "${config_path}"
done

if [[ "${RUN_CENTRALIZED}" == "1" ]]; then
  central_cmd=("${PYTHON}" -u "scripts/run_centralized_baseline.py" "--config" "${CENTRAL_CONFIG}")
  append_common_args central_cmd
  if [[ "${CENTRAL_BATCH_SIZE}" != "0" ]]; then
    central_cmd+=("--batch-size" "${CENTRAL_BATCH_SIZE}")
  fi
  if [[ "${LIMIT_ROWS}" != "0" ]]; then
    central_cmd+=("--limit-rows" "${LIMIT_ROWS}")
  fi
  run_logged "centralized_multimodal" "${central_cmd[@]}"
fi

for config_path in "${FED_CONFIGS[@]}"; do
  config_name="$(basename "${config_path}" .yaml)"

  if [[ "${RUN_FEDAVG}" == "1" ]]; then
    fedavg_cmd=("${PYTHON}" -u "scripts/run_fedavg.py" "--config" "${config_path}")
    append_common_args fedavg_cmd
    if [[ "${FED_BATCH_SIZE}" != "0" ]]; then
      fedavg_cmd+=("--batch-size" "${FED_BATCH_SIZE}")
    fi
    if [[ "${LIMIT_ROWS}" != "0" ]]; then
      fedavg_cmd+=("--max-samples-per-client" "${LIMIT_ROWS}")
    fi
    run_logged "fedavg_${config_name}" "${fedavg_cmd[@]}"
  fi

  if [[ "${RUN_FEDGATE}" == "1" ]]; then
    fedgate_cmd=("${PYTHON}" -u "scripts/run_fedgate.py" "--config" "${config_path}")
    append_common_args fedgate_cmd
    if [[ "${FED_BATCH_SIZE}" != "0" ]]; then
      fedgate_cmd+=("--batch-size" "${FED_BATCH_SIZE}")
    fi
    if [[ "${LIMIT_ROWS}" != "0" ]]; then
      fedgate_cmd+=("--max-samples-per-client" "${LIMIT_ROWS}")
    fi
    run_logged "fedgate_${config_name}" "${fedgate_cmd[@]}"
  fi
done

echo "[run_full_benchmark_matrix] logs=${LOG_DIR}"
