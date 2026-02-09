#!/bin/bash
#SBATCH -J ippo_coins
#SBATCH -A MLL
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -o /scratch/11264/rohitchawla28/scr_reproduce_socialjax/slurm/logs/ippo_coins.%j.out
#SBATCH -e /scratch/11264/rohitchawla28/scr_reproduce_socialjax/slurm/logs/ippo_coins.%j.err

set -euo pipefail

# set up paths
BASE="/scratch/11264/rohitchawla28/scr_reproduce_socialjax"
REPO="$BASE/SocialJax"
MINIFORGE="$BASE/miniforge3"

# conda
export PATH="$MINIFORGE/bin:$PATH"
source "$MINIFORGE/etc/profile.d/conda.sh"
conda activate SocialJax

# move into repo
cd "$REPO"

# added this because PYTHONPATH currently set so it might mess up imports
unset PYTHONPATH

# make sure imports work (main fix after removing the hardcoded sys.path)
export PYTHONPATH="$REPO"

# WandB local files
export WANDB_DIR="$BASE/wandb"
export WANDB_CACHE_DIR="$BASE/wandb_cache"
export WANDB_CONFIG_DIR="$BASE/wandb_config"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# will delete and make new WandB key, just need to get past the errors
export WANDB_API_KEY="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"

# other caches to make sure nothing going on $HOME
export XDG_CACHE_HOME="$BASE/.cache"
export HF_HOME="$BASE/hf"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME"

# run
python "$REPO/algorithms/IPPO/ippo_cnn_coins.py"
