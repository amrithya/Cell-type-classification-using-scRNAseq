# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import scanpy as sc
import pickle as pkl
from performer_pytorch import PerformerLM
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, default=16906)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--pos_embed", type=bool, default=True)
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad')
parser.add_argument("--model_path", type=str, default='./panglao_pretrained.pth')
parser.add_argument("--model_name", type=str, default='finetune')
args = parser.parse_args()

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
UNASSIGN_THRES = 0.0

CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed

model_name = args.model_name

dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = torch.distributed.get_world_size()

seed_all(SEED + torch.distributed.get_rank())

class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        full_seq = self.data[index].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        seq_label = self.label[index]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]

try:
    data = sc.read_h5ad(args.data_path)
    label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)
    with open('label_dict', 'wb') as fp:
        pkl.dump(label_dict, fp)
    with open('label', 'wb') as fp:
        pkl.dump(label, fp)
    label = torch.from_numpy(label)
    data = data.X
except Exception as e:
    exit(1)

sss_indices = np.arange(len(label))
train_idx = sss_indices
val_idx = sss_indices

val_dataset = SCDataset(data[val_idx], label[val_idx])
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    local_attn_heads=0,
    g2v_position_emb=POS_EMBED_USING
)

ckpt_path = f"/data1/data/corpus/scMODEL/{model_name}_full_model_Zheng68K.pkl"

try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
except Exception as e:
    exit(1)

class Identity(nn.Module):
    def __init__(self, dropout=0., h_dim=100, out_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim)

    def forward(self, x):
        x = x[:, None, :, :]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
model.eval()

net = model.module if isinstance(model, DDP) else model

for m in net.performer.modules():
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        m.register_forward_hook(lambda mod, inp, out: setattr(mod, "_x", inp[0].detach()))
        m.register_full_backward_hook(lambda mod, gi, go: setattr(mod, "_rel", (mod._x * go[0]).sum(dim=-1).detach()))

seqs = []
lbls = []
for x_v, y_v in val_loader:
    x_v = x_v.to(device)
    y_v = y_v.to(device)
    seqs.append(x_v)
    lbls.append(y_v)
    if sum(len(s) for s in seqs) >= 200:
        break
seqs = torch.cat(seqs)[:200]
lbls = torch.cat(lbls)[:200]

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
