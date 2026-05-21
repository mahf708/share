#!/bin/bash
#SBATCH -A <account>_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH -J gemma-chat
#SBATCH -o gemma-chat-%j.log

# Long-running serve. Replace <account> above with your GPU project (e.g. m9999_g).
# After it starts, look at gemma-chat-<jobid>.log for the node name + port,
# then from your laptop:
#   ssh -L 7860:<nid>:7860 perlmutter.nersc.gov
# and open http://localhost:7860

set -euo pipefail

VENV_DIR="${VENV_DIR:-$SCRATCH/envs/gemma-chat}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

module load python
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Required: Gemma is gated on HF.
: "${HF_TOKEN:?Set HF_TOKEN before sbatch (export HF_TOKEN=hf_xxx)}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME="${HF_HOME:-$SCRATCH/hf-cache}"
mkdir -p "$HF_HOME"

export GEMMA_PORT="${GEMMA_PORT:-7860}"

echo "==============================================="
echo " gemma-chat on $(hostname)"
echo " port:     $GEMMA_PORT"
echo " HF_HOME:  $HF_HOME"
echo ""
echo " From your laptop, open a tunnel:"
echo "   ssh -L ${GEMMA_PORT}:$(hostname):${GEMMA_PORT} perlmutter.nersc.gov"
echo " then browse http://localhost:${GEMMA_PORT}"
echo "==============================================="

srun --unbuffered python "$REPO_DIR/app.py"
