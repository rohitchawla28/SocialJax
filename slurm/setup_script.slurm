#!/bin/bash
#SBATCH -J socialjax_setup
#SBATCH -A MLL
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o /scratch/11264/rohitchawla28/scr_reproduce_socialjax/slurm/logs/setup.%j.out
#SBATCH -e /scratch/11264/rohitchawla28/scr_reproduce_socialjax/slurm/logs/setup.%j.err

set -euo pipefail

# set up paths
BASE="/scratch/11264/rohitchawla28/scr_reproduce_socialjax"
REPO="$BASE/SocialJax"
MINIFORGE="$BASE/miniforge3"

# conda
export PATH="$MINIFORGE/bin:$PATH"
source "$MINIFORGE/etc/profile.d/conda.sh"
conda activate SocialJax

# run from repo root so relative imports + relative output dirs work
# key thing is that it still finds socialjax package inside the REPO
cd "$REPO"

# added this because PYTHONPATH currently set to python 3.9.7 site-packages on lonestar6 so it might mess up imports
# unsetting it inside the job doesn't change it globally
unset PYTHONPATH

# make sure no cache data going on $HOME
export PIP_CACHE_DIR="$BASE/.pip_cache"
export XDG_CACHE_HOME="$BASE/.cache"
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"

# install dependencies
# pip install -r requirements.txt
conda run -n SocialJax python -m pip install -r requirements.txt

# pip install jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda run -n SocialJax python -m pip install jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# sanity checks
conda run -n SocialJax python - <<'PY'
import jax, jaxlib
print("jax:", jax.__version__)
print("jaxlib:", jaxlib.__version__)
print("devices:", jax.devices())
PY