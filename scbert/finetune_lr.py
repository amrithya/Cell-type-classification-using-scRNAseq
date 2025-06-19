# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from performer_pytorch import PerformerLM
import scanpy as sc
import pickle as pkl
from tqdm import tqdm

# ------------------ ARGUMENTS ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, default=16906)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--grad_acc", type=int, default=60)
parser.add_argument("--valid_every", type=int, default=1)
parser.add_argument("--pos_embed", type=bool, default=True)
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad')
parser.add_argument("--model_path", type=str, default='./panglao_pretrained.pth')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/')
parser.add_argument("--model_name", type=str, default='finetune')
parser.add_argument("--force_extract", action="store_true", help="Force re-extraction of embeddings even if files exist.")
args = parser.parse_args()

rank = int(os.environ["RANK"])
local_rank = args.local_rank
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

dist.init_process_group(backend='nccl')
world_size = torch.distributed.get_world_size()

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
SEQ_LEN = args.gene_num + 1
CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all(SEED + torch.distributed.get_rank())

class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0])))
        return full_seq, self.label[rand_start]

    def __len__(self):
        return self.data.shape[0]

print("Loading data...")
data = sc.read_h5ad(args.data_path)
label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)
with open('label_dict', 'wb') as fp:
    pkl.dump(label_dict, fp)
with open('label', 'wb') as fp:
    pkl.dump(label, fp)
label = torch.from_numpy(label)
data_X = data.X

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
for index_train, index_val in sss.split(data_X, label):
    data_train, label_train = data_X[index_train], label[index_train]
    data_val, label_val = data_X[index_val], label[index_val]

train_dataset = SCDataset(data_train, label_train)
val_dataset = SCDataset(data_val, label_val)

train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    local_attn_heads=0,
    g2v_position_emb=POS_EMBED_USING
)

print("Loading pretrained PerformerLM model...")
ckpt = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])

for param in model.parameters():
    param.requires_grad = False

model.to_out = nn.Identity()
model = model.to(device)
# model = DDP(model, device_ids=[local_rank], output_device=local_rank)

def extract_embeddings(model, loader):
    model.eval()
    embeddings = []
    labels_all = []
    conv = nn.Conv1d(200, 1, kernel_size=1).to(device)

    with torch.no_grad():
        for data, labels in tqdm(loader, desc="Extracting embeddings"):
            data = data.to(device)
            hidden = model(data)
            hidden = hidden.permute(0, 2, 1)
            emb = conv(hidden).squeeze(1).cpu().numpy()
            embeddings.append(emb)
            labels_all.append(labels.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    return embeddings, labels_all

save_dir = '/data1/data/corpus/'
os.makedirs(save_dir, exist_ok=True)

train_emb_path = os.path.join(save_dir, 'train_emb.npy')
train_y_path = os.path.join(save_dir, 'train_y.npy')
test_emb_path = os.path.join(save_dir, 'test_emb.npy')
test_y_path = os.path.join(save_dir, 'test_y.npy')

if not args.force_extract and os.path.exists(train_emb_path) and os.path.exists(train_y_path) \
        and os.path.exists(test_emb_path) and os.path.exists(test_y_path):
    print("Embeddings already exist. Loading from disk...")
    train_emb = np.load(train_emb_path)
    train_y = np.load(train_y_path)
    test_emb = np.load(test_emb_path)
    test_y = np.load(test_y_path)
else:
    print("Extracting embeddings...")
    train_emb, train_y = extract_embeddings(model, train_loader)
    test_emb, test_y = extract_embeddings(model, val_loader)

    np.save(train_emb_path, train_emb)
    np.save(train_y_path, train_y)
    np.save(test_emb_path, test_emb)
    np.save(test_y_path, test_y)

print(f"train_emb shape: {train_emb.shape}")
print(f"train_y shape: {train_y.shape}")
print(f"test_emb shape: {test_emb.shape}")
print(f"test_y shape: {test_y.shape}")

if local_rank == 0:
    print("Training Logistic Regression...")
    clf = LogisticRegression(penalty="l1", C=0.1, solver="liblinear")
    clf.fit(train_emb, train_y)

    pred = clf.predict(test_emb)

    acc = accuracy_score(test_y, pred)
    f1 = f1_score(test_y, pred, average='macro')
    print(f"Test Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    print(confusion_matrix(test_y, pred))
    print(classification_report(test_y, pred, target_names=label_dict.tolist(), digits=4))
