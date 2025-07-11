#!/bin/bash

#SBATCH --job-name=fix_csv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --time=24:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=results/script_results/fix_csv_%A_%a.out
#SBATCH --error=results/script_results/fix_csv_%A_%a.err
#SBATCH --array=1


poetry run python -u fix_csv.py 

echo "All Done at $(date)!"
wait
