#!/usr/bin/env bash

#SBATCH --job-name=10MSEBaby_992_15376_.6_1e-2     # Job name
#SBATCH --ntasks=1                     # Run on a single Node
#SBATCH --cpus-per-task=4
#SBATCH --mem=160gb                    # Job memory request
#SBATCH --time=96:00:00                # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --output=logs/10MSEBaby_992_15376_.6_1e-2_%j.out

eval "$(conda shell.bash hook)" 
conda activate 663FinalA
python main.py new --data-dir ../data/midjourney/ -b 2 -e 300 --name 10MSEBaby_992_15376_.6_1e-2 --size 992 -m 15376 --hash-mode None --masking-args 8 --noise 'jpeg()' --split-image --message-block-length 4