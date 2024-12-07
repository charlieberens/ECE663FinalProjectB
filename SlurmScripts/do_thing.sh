#!/usr/bin/env bash

#SBATCH --job-name=512_measurements    # Job name
#SBATCH --ntasks=1                     # Run on a single Node
#SBATCH --cpus-per-task=4
#SBATCH --mem=160gb                    # Job memory request
#SBATCH --time=96:00:00                # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=logs/512_measurements_%j.out

eval "$(conda shell.bash hook)" 
conda activate 663FinalA
python main.py continue --folder "../runs/standard-jpeg-vit-normed-frozen_.35-.025_noise-.07_coeff-9216-512 2024.12.06--02-09-45"