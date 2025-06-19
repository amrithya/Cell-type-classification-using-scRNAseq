#!/bin/bash

#SBATCH --job-name="test"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --output=results/besteffort_finetune_%A_%a.out
#SBATCH --error=results/besteffort_finetune_%A_%a.err
#SBATCH --requeue

 poetry run torchrun --nproc_per_node=4 embeddings.py

echo "All Done!"
wait
