#!/bin/bash
#SBATCH --job-name=perturbation_med
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=8:00:0
#SBATCH --output=perturbation_med.out

ml anaconda
conda activate avlm
python3 eval.py --config=configs/eval/perturbation_med.yml --device=cuda --wandb