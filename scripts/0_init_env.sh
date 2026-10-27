#!/usr/bin/env bash
# Environment recipe adapted from the original project scripts and the
# BrowseComp / BrowseComp-Plus / GPT-OSS-BrowseCompPlus-Eval references.
# This script installs dependencies only; it never starts a model or evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."

profile="${1:-client}"
case "$profile" in
  client|retrieval|server|all) ;;
  *) echo 'Usage: bash scripts/0_init_env.sh [client|retrieval|server|all]' >&2; exit 2 ;;
esac

python_bin="${PYTHON_BIN:-python3.12}"
venv_dir="${VENV_DIR:-.venv}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  "$python_bin" -m venv "$venv_dir"
  source "$venv_dir/bin/activate"
fi
python -m pip install --upgrade pip setuptools wheel

if [[ "$profile" == client || "$profile" == all ]]; then
  # qwen-agent==0.0.26 is retained from the supplied environment script.
  python -m pip install -e '.[agents]'
fi

if [[ "$profile" == retrieval || "$profile" == all ]]; then
  # Install a CUDA-compatible torch build separately if your platform needs one.
  python -m pip install -e '.[retrieval,bm25,dense]'
  python -m pip install huggingface_hub
  tevatron_dir="${TEVATRON_DIR:-external/tevatron}"
  if [[ ! -d "$tevatron_dir" ]]; then
    git clone https://github.com/texttron/tevatron.git "$tevatron_dir"
  fi
  # Optional upstream revision selection; a supplied checkout is otherwise reused.
  if [[ -n "${TEVATRON_REV:-}" ]]; then
    git -C "$tevatron_dir" checkout "$TEVATRON_REV"
  fi
  python -m pip install -e "$tevatron_dir"
  echo 'For BM25 / Lucene snippets, install Java 21 and configure JAVA_HOME.'
  echo 'Then run: python scripts/prepare_lucene_jars.py'
fi

if [[ "$profile" == server || "$profile" == all ]]; then
  # Historical serving recipe from the supplied scripts; use a separate Linux
  # CUDA environment. The client does not require a local vLLM installation.
  python -m pip install 'vllm==0.10.2' \
    --extra-index-url https://download.pytorch.org/whl/cu128
fi

echo "Installed profile: $profile. No inference or evaluation was started."
echo "For a new shell, activate the environment: source $venv_dir/bin/activate"

