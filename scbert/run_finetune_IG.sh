#!/bin/bash

#SBATCH --job-name="finetune_IG"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=results/finetune_IG_%A_%a.out
#SBATCH --error=results/finetune_IG_%A_%a.err
#SBATCH --array=1

export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

poetry run torchrun --nproc-per-node=2 finetune_IG.py \
    --data_path "/data1/data/corpus/scDATA/Zheng68K.h5ad" \
    --model_path "/data1/data/corpus/scMODEL/finetune_full_model_Zheng68K.pkl" \
    --output_dir "./IG_outputs/"

echo "All Done at $(date)!"
wait
