#!/bin/bash
# Grab one A100 interactively and launch the chat app in the foreground.
#
#   ACCOUNT=m9999_g bash interactive.sh
#
# When the shell drops into the compute node, the script auto-execs the server.

set -euo pipefail

: "${ACCOUNT:?Set ACCOUNT=<your_gpu_project>_g (e.g. m9999_g)}"
: "${HF_TOKEN:?Set HF_TOKEN=hf_xxx (Gemma is gated on HuggingFace)}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$SCRATCH/envs/gemma-chat}"
PORT="${GEMMA_PORT:-7860}"

exec salloc \
  --nodes 1 \
  --qos interactive \
  --time 02:00:00 \
  --constraint gpu \
  --gpus 1 \
  --account "$ACCOUNT" \
  bash -lc "
    module load python
    source '$VENV_DIR/bin/activate'
    export HF_TOKEN='$HF_TOKEN'
    export HUGGING_FACE_HUB_TOKEN='$HF_TOKEN'
    export HF_HOME=\"\${HF_HOME:-\$SCRATCH/hf-cache}\"
    export GEMMA_PORT='$PORT'
    echo
    echo '>>> from your laptop:  ssh -L $PORT:'\$(hostname)':$PORT perlmutter.nersc.gov'
    echo '>>> then open:         http://localhost:$PORT'
    echo
    python '$REPO_DIR/app.py'
  "
