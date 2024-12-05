#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=8:00:0
#SBATCH --output=baseline.out

ml anaconda
conda activate avlm
python3 eval.py --config=configs/eval/baseline.yml --device=cuda --wandb
