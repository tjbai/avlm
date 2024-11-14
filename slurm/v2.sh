#!/bin/bash
#SBATCH --job-name=v2
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=8:00:0
#SBATCH --output=v2.out

ml anaconda
conda activate avlm
python3 train.py --config=configs/patch/v2.yml --device=cuda --wandb
