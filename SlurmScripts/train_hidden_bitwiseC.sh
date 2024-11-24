#!/usr/bin/env bash

#SBATCH --job-name=bitwiseC_midjourney_6_256_jpeg     # Job name
#SBATCH --ntasks=1                     # Run on a single Node
#SBATCH --cpus-per-task=4
#SBATCH --mem=160gb                    # Job memory request
#SBATCH --time=96:00:00                # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --output=logs/bitwiseC_midjourney_6_256_jpeg_%j.out

eval "$(conda shell.bash hook)" 
conda activate 663FinalA
python main.py new --data-dir ../data/midjourney/ -b 6 -e 300 --name bitwiseC_midjourney_5_256_jpeg --size 512 -m 200 --hash-mode bitwiseC --masking-args 6 --noise 'jpeg()'
