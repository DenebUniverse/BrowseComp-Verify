#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Download only the corpus and the selected retrieval index; no model weights.
index="${1:-qwen3-embedding-8b}"
case "$index" in
  bm25|qwen3-embedding-0.6b|qwen3-embedding-4b|qwen3-embedding-8b) ;;
  *) echo 'Choose bm25 or qwen3-embedding-{0.6b,4b,8b}' >&2; exit 2 ;;
esac
hf download Tevatron/browsecomp-plus-corpus --repo-type dataset \
  --include 'data/*.parquet' --local-dir external/browsecomp-plus-corpus
hf download Tevatron/browsecomp-plus-indexes --repo-type dataset \
  --include "$index/*" --local-dir external/indexes

