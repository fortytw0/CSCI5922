#!/bin/bash
​
#SBATCH --nodes=1
#SBATCH --time=03:00:00
##SBATCH --partition=sgpu
#SBATCH --partition=sgpu-testing
#SBATCH --ntasks=24
#SBATCH --job-name=gpu-job
#SBATCH --output=gpu-csci5922.%j.out

​module purge
module load cuda
module load cudnn
​
nvidia-smi

conda activate csci5922

python -m src.rnn-deep-learning.py