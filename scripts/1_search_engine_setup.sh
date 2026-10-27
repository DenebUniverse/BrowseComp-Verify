#!/usr/bin/env bash
# Start the configured fixed-corpus retrieval service after loading .env.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/start_search.sh "$@"

