#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CORPUS_PARQUET_PATH:?Set CORPUS_PARQUET_PATH (see .env.example)}"
export SEARCHER_TYPE="${SEARCHER_TYPE:-dense}"
if [[ "$SEARCHER_TYPE" == "dense" ]]; then
  : "${DENSE_INDEX_PATH:?Set DENSE_INDEX_PATH}"
  : "${DENSE_MODEL_NAME:?Set DENSE_MODEL_NAME}"
elif [[ "$SEARCHER_TYPE" == "bm25" ]]; then
  : "${LUCENE_INDEX_DIR:?Set LUCENE_INDEX_DIR}"
else
  echo 'SEARCHER_TYPE must be dense or bm25' >&2
  exit 2
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec python -m uvicorn browsecomp_verify.retrieval.service:app \
  --host "${SEARCH_HOST:-127.0.0.1}" --port "${SEARCH_PORT:-8000}" --workers 1

