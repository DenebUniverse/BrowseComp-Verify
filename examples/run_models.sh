#!/usr/bin/env bash
# Editable single-model examples. Executing this script makes model API calls.
# Configure your own services and keys; no private endpoints are embedded here.
set -euo pipefail
cd "$(dirname "$0")/.."

agent="${1:?Choose claude, qwen3_5, deepseek_v4_flash, deepseek_v4_pro, or deepseek_v4_flash_visit}"
shift
turns=100
effort=high
summary_args=()
case "$agent" in
  claude) model=claude-opus-4-6-cc ;;
  qwen3_5) model=Qwen3.5-397B-A17B ;;
  deepseek_v4_flash) model=deepseek-v4-flash; effort=max ;;
  deepseek_v4_pro) model=deepseek-v4-pro; effort=max ;;
  deepseek_v4_flash_visit) model=deepseek-v4-flash; effort=max; turns=500 ;;
  *) echo 'Unknown example interface' >&2; exit 2 ;;
esac
if [[ "$agent" == qwen3_5 || "$agent" == deepseek_v4_flash_visit ]]; then
  : "${SUMMARY_URL:?Set the separate summary-model endpoint}"
  summary_args=(--summary_url "$SUMMARY_URL")
fi
: "${MODEL_URL:?Set your compatible model endpoint}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m browsecomp_verify.run \
  --agent "$agent" --task verify \
  --model_name "${MODEL_NAME:-$model}" \
  --model_url "$MODEL_URL" \
  --search_url "${SEARCH_URL:-http://127.0.0.1:8000}" \
  --data_path data/diamond.jsonl \
  --output_dir "${OUTPUT_DIR:-results/diamond-$agent-strict}" \
  --prompt strict --reasoning_effort "$effort" \
  --max_turns "${MAX_TURNS:-$turns}" --num_workers "${NUM_WORKERS:-1}" \
  "${summary_args[@]}" "$@"
