#!/bin/bash
#SBATCH --job-name=patch_large_downscaled
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=8:00:0
#SBATCH --output=patch_large_downscaled.out

ml anaconda
conda activate avlm
python3 eval.py --config=configs/eval/patch_large_downscaled.yml --device=cuda --wandb
