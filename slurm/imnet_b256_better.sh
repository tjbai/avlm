#!/bin/bash
#SBATCH --job-name=imnet_b256_better
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=8:00:0
#SBATCH --output=imnet_b256_better.out

ml anaconda
conda activate avlm
python3 train.py --config=configs/imnet_b256_better.yml --device=cuda --wandb
