#!/bin/bash
#SBATCH --job-name=patch_med_upscaled
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=8:00:0
#SBATCH --output=patch_med_upscaled.out

ml anaconda
conda activate avlm
python3 eval.py --config=configs/eval/patch_med_upscaled.yml --device=cuda --wandb
