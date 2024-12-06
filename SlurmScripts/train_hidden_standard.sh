#!/usr/bin/env bash

#SBATCH --job-name=standard-jpeg-vit-normed-frozen_.7-.025_noise-.07_coeff-11532-992     # Job name
#SBATCH --ntasks=1                     # Run on a single Node
#SBATCH --cpus-per-task=4
#SBATCH --mem=160gb                    # Job memory request
#SBATCH --time=96:00:00                # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --output=logs/modern/standard-jpeg-vit-normed-frozen_.7-.025_noise-.07_coeff-11532-992-%j.out

eval "$(conda shell.bash hook)" 
conda activate 663FinalA
python main.py new --data-dir ../data/midjourney/ -b 2 -e 300 --name standard-jpeg-vit-normed-frozen_.7-.025_noise-.07_coeff-11532-992 --size 992 -m 11532 --hash-mode screen --masking-args 1 --split-image --message-block-length 3 --encoder-coeff .7 --noise 'jpeg()'