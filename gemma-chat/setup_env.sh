#!/bin/bash
# One-time env setup. Run on a Perlmutter login node.
#
#   bash setup_env.sh
#
# Creates a uv-managed venv under $SCRATCH/envs/gemma-chat and installs deps.
# Re-run safely; uv will no-op when nothing changed.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$SCRATCH/envs/gemma-chat}"

module load python

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p "$(dirname "$VENV_DIR")"
if [[ ! -d "$VENV_DIR" ]]; then
  uv venv --python 3.11 "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
uv pip install -r "$REPO_DIR/requirements.txt"

cat <<EOF

[setup] done.
  venv:    $VENV_DIR
  activate: source $VENV_DIR/bin/activate

Next:
  1) export HF_TOKEN=hf_xxx           # required: Gemma is gated
  2) sbatch $REPO_DIR/serve.sbatch    # or: bash $REPO_DIR/interactive.sh
EOF
