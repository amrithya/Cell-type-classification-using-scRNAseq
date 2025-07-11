#!/bin/bash

#SBATCH --job-name="finetune_deeplift"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=results/finetune_deeplift_%A_%a.out
#SBATCH --error=results/finetune_deeplift_%A_%a.err
#SBATCH --array=1

export CUDA_LAUNCH_BLOCKING=1

poetry run python -u finetune_deeplift.py \
    --data_path "/data1/data/corpus/scDATA/Zheng68K.h5ad" \
    --model_path "/data1/data/corpus/scMODEL/panglao_pretrain.pth"

echo "All Done at $(date)!"
wait
