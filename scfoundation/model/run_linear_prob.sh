#!/bin/bash

#SBATCH --job-name=linear_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --time=24:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=results/script_results/linear_probe_%j.out
#SBATCH --error=results/script_results/linear_probe_%j.err

python -u linear_probing_classifier.py

echo "All Done at $(date)!"
wait
