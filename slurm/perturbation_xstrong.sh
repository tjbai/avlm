#!/bin/bash
#SBATCH --job-name=perturbation_xstrong
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=8:00:0
#SBATCH --output=perturbation_xstrong.out

ml anaconda
conda activate avlm
python3 train.py --config=configs/perturbation/perturbation_xstrong.yml --device=cuda --wandb
