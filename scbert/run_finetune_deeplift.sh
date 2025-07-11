#!/bin/bash

#SBATCH --job-name=deeplift_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=results/deeplift_eval_%j.out
#SBATCH --error=results/deeplift_eval_%j.err

export CUDA_LAUNCH_BLOCKING=1

poetry run python -u finetune_deeplift.py \
    --data_path "/data1/data/corpus/scDATA/Zheng68K.h5ad" \
    --model_path "/data1/data/corpus/scMODEL/panglao_pretrained.pth" \
    --output_dir "./deeplift_outputs/"

echo "All Done at $(date)!"
