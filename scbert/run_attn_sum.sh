#!/bin/bash

#SBATCH --job-name="attn_sum_save"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=results/attn_sum_save_%A_%a.out
#SBATCH --error=results/attn_sum_save_%A_%a.err
#SBATCH --array=1

poetry run python -u attn_sum_save.py \
    --data_path "/data1/data/corpus/scDATA/Zheng68K.h5ad" \
    --model_path "/data1/data/corpus/scMODEL/finetune_full_model_Zheng68K.pkl" \
    --save_dir "/data1/data/corpus/scDATA/"

echo "All Done at $(date)!"
wait
