#!/bin/bash

#SBATCH --job-name="cancer_finetune"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --time=24:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=results/cancer_finetune_%A_%a.out
#SBATCH --error=results/cancer_finetune_%A_%a.err
#SBATCH --array=1


export CUDA_LAUNCH_BLOCKING=1

poetry run python -m torch.distributed.launch --nproc_per_node=1 finetune_cancer.py \
    --data_path "/data1/data/corpus/scDATA/cancer/data/mt_kidney_rcc_cca_paa_2/grade/preprocessed_data_mt_kidney_rcc_cca_paa_2.h5ad" \
    --model_path "/data1/data/corpus/scMODEL/panglao_pretrain.pth"

echo "All Done at $(date)!"
wait
