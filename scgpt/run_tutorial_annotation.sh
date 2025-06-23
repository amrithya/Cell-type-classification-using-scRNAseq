#!/bin/bash

#SBATCH --job-name="tutorial_annotation"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --time=24:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=results/tutorial_annotation_%A_%a.out
#SBATCH --error=results/tutorial_annotation_%A_%a.err
#SBATCH --array=1


torchrun --nproc_per_node=4 tutorials/tutorial_annotation.py

echo "All Done!"
wait
