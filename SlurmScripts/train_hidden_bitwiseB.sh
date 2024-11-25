#!/usr/bin/env bash

#SBATCH --job-name=midjourney_30_None_128_jpeg_fixed_     # Job name
#SBATCH --ntasks=1                     # Run on a single Node
#SBATCH --cpus-per-task=4
#SBATCH --mem=160gb                    # Job memory request
#SBATCH --time=96:00:00                # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --output=logs/midjourney_30_None_128_jpeg_fixed_%j.out

eval "$(conda shell.bash hook)" 
conda activate 663FinalA
python main.py new --data-dir ../data/midjourney/ -b 12 -e 300 --name midjourney_60_None_128_jpeg_fixed --size 128 -m 30 --hash-mode None --masking-args 8 --noise 'jpeg()'