#!/bin/bash
#SBATCH -J jax_test
#SBATCH -A MLL
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2               # good for runtime orchestration?
#SBATCH -t 00:05:00
#SBATCH -o /scratch/11264/rohitchawla28/scr_reproduce_socialjax/slurm/logs/jax_test.%j.out
#SBATCH -e /scratch/11264/rohitchawla28/scr_reproduce_socialjax/slurm/logs/jax_test.%j.err

# restart environment modules
module reset

# make cuda available in this shell
module load cuda

# set path
export PATH="/scratch/11264/rohitchawla28/scr_reproduce_socialjax/miniforge3/bin:$PATH"

# check gpu
nvidia-smi

# test
conda run -n SocialJax python -c "import jax; print(jax.devices())"