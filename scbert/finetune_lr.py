# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
import joblib
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *
import pickle as pkl
from tqdm import tqdm
import shap
from sklearn.linear_model import LogisticRegression

# -- utils functions placeholders (you should keep your original utils.py functions here) --
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
parser.add_argument("--gene_num", type=int, default=16906)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad')
parser.add_argument("--model_path", type=str, default='./panglao_pretrained.pth')
parser.add_argument("--model_name", type=str, default='scbert_finetune')
args = parser.parse_args()

rank = int(os.environ.get("RANK", 0))
local_rank = args.local_rank
is_master = (local_rank == 0)

SEED = args.seed
BATCH_SIZE = args.batch_size
SEQ_LEN = args.gene_num + 1
model_name = args.model_name

dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = dist.get_world_size()

seed_all(SEED + rank)

if is_master:
    print(f"[Init] Seed: {SEED}, Batch size: {BATCH_SIZE}")
    print(f"[Init] Using {world_size} GPUs, local_rank: {local_rank}")

class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        full_seq = self.data[index].toarray()[0]
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        seq_label = self.label[index]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]

try:
    if is_master:
        print("Loading data...")
    data = sc.read_h5ad(args.data_path)
    label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)
    if is_master:
        with open('label_dict', 'wb') as fp:
            pkl.dump(label_dict, fp)
        with open('label', 'wb') as fp:
            pkl.dump(label, fp)
    label = torch.from_numpy(label)
    data = data.X
except Exception as e:
    if is_master:
        print(f"[ERROR] Data loading failed: {e}")
    exit(1)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
for index_train, index_val in sss.split(data, label):
    data_train, label_train = data[index_train], label[index_train]
    data_val, label_val = data[index_val], label[index_val]

train_dataset = SCDataset(data_train, label_train)
val_dataset = SCDataset(data_val, label_val)

train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

model = PerformerLM(
    num_tokens=7,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    local_attn_heads=0,
    g2v_position_emb=True
)

try:
    if is_master:
        print("Loading pretrained PerformerLM model...")
    ckpt = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
except Exception as e:
    if is_master:
        print(f"[ERROR] Model loading failed: {e}")
    exit(1)

model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
model.eval()

for param in model.parameters():
    param.requires_grad = False

def extract_cls_embeddings(model, dataloader, device):
    embeddings = []
    labels = []
    for data, label in tqdm(dataloader, desc="Extracting embeddings"):
        data = data.to(device)
        print("Input shape before model:", data.shape)
        print(data.dtype)
        with torch.no_grad():
            out = model.module.performer(data) 
        print("Output shape after model:", out.shape)
        cls_emb = out[:, 0, :].cpu().numpy()
        embeddings.append(cls_emb)
        labels.append(label.numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    embeddings_tensor = torch.tensor(embeddings).to(device)
    labels_tensor = torch.tensor(labels).to(device)
    embeddings_all = distributed_concat(embeddings_tensor, len(dataloader.sampler.dataset), world_size).cpu().numpy()
    labels_all = distributed_concat(labels_tensor, len(dataloader.sampler.dataset), world_size).cpu().numpy()
    return embeddings_all, labels_all

if is_master:
    print("[INFO] Extracting embeddings for training set")
train_embeddings, train_labels = extract_cls_embeddings(model, train_loader, device)

if is_master:
    print("[INFO] Extracting embeddings for validation set")
val_embeddings, val_labels = extract_cls_embeddings(model, val_loader, device)

if is_master:
    print("[INFO] Training LogisticRegression classifier")
    logreg = LogisticRegression(
        max_iter=1000, multi_class='multinomial', solver='lbfgs', n_jobs=-1
    )
    logreg.fit(train_embeddings, train_labels)

    val_preds = logreg.predict(val_embeddings)
    acc = accuracy_score(val_labels, val_preds)
    print(f"[LOGREG] Validation Accuracy: {acc:.4f}")
    print("[LOGREG] Classification Report:")
    print(classification_report(val_labels, val_preds, digits=4))

    os.makedirs('logreg_models', exist_ok=True)
    logreg_path = os.path.join('logreg_models', f'{model_name}_logreg.pkl')
    joblib.dump(logreg, logreg_path)
    print(f"[LOGREG] LogisticRegression model saved to {logreg_path}")
