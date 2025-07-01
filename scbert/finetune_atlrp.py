# -*- coding: utf-8 -*-
import os
import gc
import argparse
import pickle
import json
import random
import math
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch
import shap
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from performer_pytorch import PerformerLM
import scanpy as sc
from utils import *
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, default=16906)
parser.add_argument("--epoch", type=int, default=21)
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
args = parser.parse_args()

rank = int(os.environ["RANK"])
local_rank = args.local_rank
is_master = local_rank == 0

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every

PATIENCE = 10
CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed
model_name = args.model_name
ckpt_path = f"/data1/data/corpus/scMODEL/{model_name}_model_Zheng68K.pkl"

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = dist.get_world_size()

seed_all(SEED + rank)

if is_master:
    print(f"Init: seed={SEED}, epochs={EPOCHS}, bs={BATCH_SIZE}, lr={LEARNING_RATE}, gpus={world_size}")

class SCDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        arr = self.data[idx].toarray()[0]
        arr[arr > CLASS-2] = CLASS-2
        seq = np.concatenate([arr, [0]]).astype(int)
        return torch.from_numpy(seq).long(), torch.tensor(self.label[idx], dtype=torch.long)

class IdentityHead(nn.Module):
    def __init__(self, seq_len, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, (1, 200))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(seq_len, num_classes)
        self.act = nn.ReLU()
        self._printed = False
    def forward(self, x):
        if not self._printed:
            print("Before head:", x.shape)
        x = x.unsqueeze(1)
        x = self.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.act(self.fc(x))
        if not self._printed:
            print("After head:", x.shape)
            self._printed = True
        return x

adata = sc.read_h5ad(args.data_path)
label_dict, labels = np.unique(adata.obs['celltype'], return_inverse=True)
with open('label_dict.pkl', 'wb') as f: pickle.dump(label_dict, f)
with open('labels.pkl', 'wb') as f: pickle.dump(labels, f)
data = adata.X

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, val_idx = next(sss.split(data, labels))
train_ds = SCDataset(data[train_idx], labels[train_idx])
val_ds = SCDataset(data[val_idx], labels[val_idx])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=DistributedSampler(train_ds))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, sampler=DistributedSampler(val_ds))

model = PerformerLM(num_tokens=CLASS, dim=200, depth=6, max_seq_len=SEQ_LEN, heads=10, local_attn_heads=0, g2v_position_emb=POS_EMBED_USING)
ckpt = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])

for p in model.parameters(): p.requires_grad = False
for p in model.norm.parameters(): p.requires_grad = True
for p in model.performer.net.layers[-2].parameters(): p.requires_grad = True

model.to_out = IdentityHead(SEQ_LEN, len(label_dict))
model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

optim = Adam(model.parameters(), lr=LEARNING_RATE)
sched = CosineAnnealingWarmupRestarts(optim, first_cycle_steps=15, cycle_mult=2, max_lr=LEARNING_RATE, min_lr=1e-6, warmup_steps=5, gamma=0.9)
loss_fn = nn.CrossEntropyLoss().to(device)

start_epoch = 1; best_acc = 0; stale = 0
if os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu')
    start_epoch = ck.get('epoch', 0) + 1
    best_acc = ck.get('best_acc', 0)
    model.load_state_dict(ck['model_state_dict'])
    optim.load_state_dict(ck['optimizer_state_dict'])
    sched.load_state_dict(ck['scheduler_state_dict'])
    if is_master:
        print(f"Resumed from epoch {start_epoch} with acc={best_acc:.4f}")

for epoch in range(start_epoch, EPOCHS+1):
    model.train()
    train_loader.sampler.set_epoch(epoch)
    rl = 0; ra = 0
    for i, (x, y) in enumerate(tqdm(train_loader, desc=f"Train {epoch}")):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        if (i + 1) % GRADIENT_ACCUMULATION == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
            optim.step(); optim.zero_grad()
        rl += loss.item()
        ra += (logits.argmax(-1) == y).float().mean().item()
    train_loss = get_reduced(rl / (i + 1), local_rank, 0, world_size)
    train_acc = get_reduced(ra / (i + 1) * 100, local_rank, 0, world_size)
    if is_master:
        print(f"[Epoch {epoch}] Train loss={train_loss:.4f}, acc={train_acc:.2f}%")
    sched.step(); dist.barrier()

    if epoch % VALIDATE_EVERY == 0:
        model.eval()
        vrl = 0; preds = []; trues = []
        with torch.no_grad():
            for x_v, y_v in tqdm(val_loader, desc=f"Valid {epoch}"):
                x_v = x_v.to(device)
                y_v = y_v.to(device)
                logv = model(x_v)
                vrl += loss_fn(logv, y_v).item()
                preds.append(logv.argmax(-1).cpu())
                trues.append(y_v.cpu())
        vrl /= len(val_loader)
        if is_master: print("Val loss:", vrl)
        preds = torch.cat(preds); trues = torch.cat(trues)
        mask = preds >= 0
        acc_v = accuracy_score(trues[mask], preds[mask])
        f1_v = f1_score(trues[mask], preds[mask], average='macro')
        if is_master:
            print(confusion_matrix(trues[mask], preds[mask]))
            print(classification_report(trues[mask], preds[mask], target_names=label_dict, digits=4))
        if acc_v > best_acc:
            best_acc = acc_v; stale = 0
            save_ckpt(epoch, model, optim, sched, vrl, model_name)
        else:
            stale += 1
            if stale > PATIENCE:
                if is_master: print("Early stopping"); break

if is_master:
    print("\nComputing relevance per class...")
model.eval()
net = model.module if isinstance(model, DDP) else model

for m in net.performer.modules():
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        m.register_forward_hook(lambda mod, inp, out: setattr(mod, "_x", inp[0].detach()))
        m.register_full_backward_hook(lambda mod, gi, go: setattr(mod, "_rel", (mod._x * go[0]).sum(dim=-1).detach()))

seqs = []; lbls = []
for x_v, y_v in val_loader:
    x_v = x_v.to(device)
    y_v = y_v.to(device)
    seqs.append(x_v); lbls.append(y_v)
    if sum(len(s) for s in seqs) >= 200: break
seqs = torch.cat(seqs)[:200]; lbls = torch.cat(lbls)[:200]

relevances = []
for seq, lbl in zip(seqs, lbls):
    net.zero_grad(set_to_none=True)
    reps = net.to_tokens(seq.unsqueeze(0)).clone().detach().requires_grad_(True)
    logits = net.forward_tokens(reps)
    logits[0, lbl].backward()
    rel = (reps.grad * reps).sum(dim=-1).squeeze().cpu().numpy()
    rel = rel / (np.max(np.abs(rel)) + 1e-12)
    relevances.append(rel)
relevances = np.stack(relevances)[:, :-1]

records = []
for cls in np.unique(lbls.cpu()):
    arr = np.mean(np.abs(relevances[lbls.cpu() == cls]), axis=0)
    top15 = arr.argsort()[-15:][::-1]
    bot15 = arr.argsort()[:15]
    for rank, g in enumerate(top15, 1):
        records.append((label_dict[cls], rank, g, arr[g]))
    for rank, g in enumerate(bot15[::-1], 1):
        records.append((label_dict[cls], -rank, g, arr[g]))

df = pd.DataFrame(records, columns=['class', 'rank', 'gene', 'value'])
df.to_csv(f"{model_name}_relevance_topbot.csv", index=False)
if is_master:
    print("Saved:", f"{model_name}_relevance_topbot.csv")
