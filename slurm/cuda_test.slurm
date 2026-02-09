#!/bin/bash
#SBATCH -J cuda_test
#SBATCH -A MLL
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2               # good for runtime orchestration?
#SBATCH -t 00:05:00
#SBATCH -o /scratch/11264/rohitchawla28/scr_reproduce_socialjax/slurm/logs/cuda_test.%j.out
#SBATCH -e /scratch/11264/rohitchawla28/scr_reproduce_socialjax/slurm/logs/cuda_test.%j.err

# restart environment modules
module reset

# see what cuda modules are available on lonestar6
module avail cuda

# make cuda available in this shell
module load cuda/11.8

# check cuda version, || true b/c some CUDA modules provide runtime libraries without nvcc which is fine for JAX
nvcc --version || true

# set path
export PATH="/scratch/11264/rohitchawla28/scr_reproduce_socialjax/miniforge3/bin:$PATH"

# check gpu
nvidia-smi

# check  visibility
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# test
conda run -n SocialJax python -c "import jax; print(jax.devices())"