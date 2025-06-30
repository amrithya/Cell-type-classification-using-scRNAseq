#!/bin/bash

#SBATCH --job-name="finetune_shap"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=3
#SBATCH --time=24:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=results/finetune_shap_%A_%a.out
#SBATCH --error=results/finetune_shap_%A_%a.err
#SBATCH --array=1


export CUDA_LAUNCH_BLOCKING=1

poetry run python -m torch.distributed.launch --nproc_per_node=2 finetune_shap.py \
    --data_path "/data1/data/corpus/scDATA/Zheng68K.h5ad" \
    --model_path "/data1/data/corpus/scMODEL/panglao_pretrain.pth"

echo "All Done at $(date)!"
wait
