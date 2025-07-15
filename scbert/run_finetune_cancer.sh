#!/bin/bash

#SBATCH --job-name="cancer_finetune"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=3
#SBATCH --time=24:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=results/cancer_finetune_%A_%a.out
#SBATCH --error=results/cancer_finetune_%A_%a.err
#SBATCH --array=1


export CUDA_LAUNCH_BLOCKING=1

poetry run python -m torch.distributed.launch --nproc_per_node=2 finetune_cancer.py \
    --data_path "/data1/data/corpus/scDATA/cancer/data/pc_UterusG_eac/preprocessed_data_pc_UterusG_eac_bert.h5ad" \
    --model_path "/data1/data/corpus/scMODEL/panglao_pretrain.pth"

echo "All Done at $(date)!"
wait
