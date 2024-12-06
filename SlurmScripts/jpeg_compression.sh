#!/usr/bin/env bash

#SBATCH --job-name=compression    # Job name
#SBATCH --ntasks=1                     # Run on a single Node
#SBATCH --cpus-per-task=4
#SBATCH --mem=160gb                    # Job memory request
#SBATCH --time=96:00:00                # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --output=logs/compression-%j.out

eval "$(conda shell.bash hook)" 
conda activate 663FinalA
python train_jpeg_compressor.py ../data/midjourney