#!/usr/bin/env bash

#SBATCH --job-name=bitwise_b_noisy_512_1024_4_     # Job name
#SBATCH --ntasks=1                     # Run on a single Node
#SBATCH --cpus-per-task=4
#SBATCH --mem=160gb                    # Job memory request
#SBATCH --time=96:00:00                # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --output=logs/bitwise_b_noisy_512_1024_4_%j.out

eval "$(conda shell.bash hook)" 
conda activate 663FinalA
python main.py new --data-dir ../data/midjourney/ -b 8 -e 300 --name bitwise_b_512_1024_4 --size 512 -m 1024 --hash-mode bitwiseC --masking-args 4 --split-image --message-block-length 1
