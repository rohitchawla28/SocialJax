#!/bin/bash
#SBATCH -J cuda_test
#SBATCH -A MLL
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2               # good for runtime orchestration?
#SBATCH --gpus=1
#SBATCH -t 00:05:00
#SBATCH -o /scratch/11264/rohitchawla28/scr_reproduce_socialjax/slurm/logs/cuda_test2.%j.out
#SBATCH -e /scratch/11264/rohitchawla28/scr_reproduce_socialjax/slurm/logs/cuda_test2.%j.err

# restart environment modules
module reset

# load cuda 12 module
module load cuda/12.2

# check cuda version, || true b/c some CUDA modules provide runtime libraries without nvcc which is fine for JAX
nvcc --version || true

# so that conda is found
export PATH="/scratch/11264/rohitchawla28/scr_reproduce_socialjax/miniforge3/bin:$PATH"

# install jax with cuda 12 support
conda run -n SocialJax python -m pip uninstall -y jax jaxlib
conda run -n SocialJax python -m pip install -U "jax[cuda12]"

# check gpu
nvidia-smi

# check  visibility
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# test
conda run -n SocialJax python -c "import jax; print(jax.devices())"