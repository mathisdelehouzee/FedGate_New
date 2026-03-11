#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

if [ -x "FedGate/.venv/bin/python" ]; then
  PY="FedGate/.venv/bin/python"
else
  PY="python3"
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="fedgate_full/logs"
METHODS_CSV="${METHODS_CSV:-centralized,fedavg_concat,fedavg_mean,fedgate,ditto_concat_global}"
PREPARE_SPLITS="${PREPARE_SPLITS:-1}"
EXPORT_BENCHMARK="${EXPORT_BENCHMARK:-1}"

mkdir -p "${LOG_DIR}"

CONFIGS=(
  "fedgate_full/configs/s0_congruent_iid.yaml"
  "fedgate_full/configs/s1_congruent_non_iid.yaml"
  "fedgate_full/configs/s2_non_congruent_iid.yaml"
  "fedgate_full/configs/s3_non_congruent_non_iid.yaml"
)

run_cfg() {
  local cfg="$1"
  local tag
  local log
  local artifacts_root

  tag="$(basename "${cfg}" .yaml)"
  log="${LOG_DIR}/${tag}_${RUN_ID}.log"

  echo "[START] ${tag} cfg=${cfg}" | tee -a "${log}"
  echo "[METHODS] ${METHODS_CSV}" | tee -a "${log}"

  if [ "${PREPARE_SPLITS}" = "1" ]; then
    artifacts_root="$(${PY} - "${cfg}" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1]).resolve()
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
art_root = Path(cfg["paths"]["artifacts_root"])
if not art_root.is_absolute():
    art_root = (cfg_path.parent / art_root).resolve()
print(art_root)
PY
)"
    ${PY} result_paper/scripts/build_splits.py --config "${cfg}" |& tee -a "${log}"
    ${PY} result_paper/scripts/inspect_splits.py \
      --splits "${artifacts_root}/splits/splits_manifest.json" \
      --output "${artifacts_root}/splits/splits_report.json" \
      --fail-on-error |& tee -a "${log}"
  fi

  IFS=',' read -r -a methods <<< "${METHODS_CSV}"
  local ditto_ran=0
  for raw_method in "${methods[@]}"; do
    local method
    method="$(printf '%s' "${raw_method}" | xargs)"
    case "${method}" in
      centralized)
        ${PY} result_paper/scripts/run_centralized.py --config "${cfg}" |& tee -a "${log}"
        ;;
      fedavg)
        ${PY} result_paper/scripts/run_fedavg.py --config "${cfg}" |& tee -a "${log}"
        ;;
      fedavg_mean)
        ${PY} result_paper/scripts/run_fedavg_mean.py --config "${cfg}" |& tee -a "${log}"
        ;;
      fedavg_concat)
        ${PY} result_paper/scripts/run_fedavg_concat.py --config "${cfg}" |& tee -a "${log}"
        ;;
      fedgate)
        ${PY} result_paper/scripts/run_fedgate.py --config "${cfg}" |& tee -a "${log}"
        ;;
      ditto_concat|ditto_concat_global|ditto_concat_personal)
        if [ "${ditto_ran}" = "0" ]; then
          ${PY} result_paper/scripts/run_ditto_concat.py --config "${cfg}" |& tee -a "${log}"
          ditto_ran=1
        fi
        ;;
      "")
        ;;
      *)
        echo "[WARN] Unknown method '${method}'" | tee -a "${log}"
        ;;
    esac
  done

  echo "[END] ${tag} exit=0" | tee -a "${log}"
}

if [ "$#" -gt 0 ]; then
  CONFIGS=("$@")
fi

for cfg in "${CONFIGS[@]}"; do
  run_cfg "${cfg}"
done

if [ "${EXPORT_BENCHMARK}" = "1" ]; then
  ${PY} fedgate_full/scripts/export_all_method_benchmarks.py \
    --results-root "fedgate_full/results" \
    --output-dir "fedgate_full/results/paper_assets_all_methods" \
    --configs "${CONFIGS[@]}"
fi
