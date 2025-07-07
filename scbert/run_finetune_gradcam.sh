#!/bin/bash

#SBATCH --job-name="finetune_gradcam"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=results/finetune_gradcam_%A_%a.out
#SBATCH --error=results/finetune_gradcam_%A_%a.err
#SBATCH --array=1

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

export CUDA_LAUNCH_BLOCKING=1

poetry run python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=$MASTER_PORT finetune_gradcam.py \
    --data_path "/data1/data/corpus/scDATA/Zheng68K.h5ad" \
    --model_path "/data1/data/corpus/scMODEL/panglao_pretrain.pth"

echo "All Done at $(date)!"
wait
