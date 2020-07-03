#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=m1759
#SBATCH --qos=special
# -t and -n should be the same
#SBATCH -t 16
#SBATCH -n 16
#SBATCH --job-name=drl_test
#SBATCH --mail-user=zulissi@andrew.cmu.edu
#SBATCH --time=02:00:00

#module load ffmpeg
export OMP_NUM_THREADS=1
conda activate surface_seg
alias ffmpeg="ffmpeg -threads 0"
srun -n 1 -c 16 python gym_trpo_parallel_training.py
