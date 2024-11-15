#!/bin/bash
#SBATCH --job-name=v4
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=8:00:0
#SBATCH --output=v4.out

ml anaconda
conda activate avlm
python3 train.py --config=configs/patch/v4.yml --device=cuda --wandb
