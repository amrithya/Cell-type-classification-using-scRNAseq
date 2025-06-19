#!/bin/bash

#SBATCH --job-name="scBERT_finetune_lr"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --time=24:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=results/scBERT_finetune_lr_%A_%a.out
#SBATCH --error=results/scBERT_finetune_lr_%A_%a.err
#SBATCH --array=1


poetry run python -m torch.distributed.launch --nproc_per_node=1 finetune_lr.py \
    --force_extract \
    --data_path "/data1/data/corpus/scDATA/Zheng68K.h5ad" \
    --model_path "/data1/data/corpus/scMODEL/panglao_pretrain.pth"
    

echo "All Done at $(date)!"
wait
