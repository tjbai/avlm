#!/bin/bash
#SBATCH --job-name=toy
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=8:00:0
#SBATCH --output=toy.out

ml anaconda
conda activate avlm
python3 eval.py --config=configs/eval/toy.yml --device=cuda --wandb
