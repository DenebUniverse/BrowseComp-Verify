#!/usr/bin/env bash
# EXAMPLE: GPT-OSS verification on DIAMOND.
# Edit these model/endpoint/output settings for your own deployment.
# Running this example calls the configured model and may incur API charges.
# No model weights, model server, or retrieval server are started by this script.
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m browsecomp_verify.run \
  --agent gptoss \
  --task verify \
  --model_name "${MODEL_NAME:-gpt-oss-120b}" \
  --model_url "${MODEL_URL:-http://127.0.0.1:8001/v1}" \
  --search_url "${SEARCH_URL:-http://127.0.0.1:8000}" \
  --data_path data/diamond.jsonl \
  --output_dir "${OUTPUT_DIR:-results/diamond-gptoss-strict}" \
  --prompt strict \
  --reasoning_effort high \
  --max_turns 100 \
  --num_workers "${NUM_WORKERS:-1}" \
  "$@"

