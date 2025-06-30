#!/bin/bash

#SBATCH --job-name="finetune_shap"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=3
#SBATCH --time=24:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=results/finetune_shap_%A.out
#SBATCH --error=results/finetune_shap_%A.err

export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1


MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

poetry run torchrun --nproc_per_node=2 --master_port=$MASTER_PORT finetune_shap.py \
    --data_path "/data1/data/corpus/scDATA/Zheng68K.h5ad" \
    --model_path "/data1/data/corpus/scMODEL/panglao_pretrain.pth"

echo "All Done at $(date)!"
wait
