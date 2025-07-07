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
from utils import *
from performer_pytorch import PerformerLM
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, default=16906)
parser.add_argument("--epoch", type=int, default=20)
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
UNASSIGN_THRES = 0.0

CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed
model_name = args.model_name

dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = dist.get_world_size()

seed_all(SEED + dist.get_rank())

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
    adata = sc.read_h5ad(args.data_path)
    gene_names = list(adata.var_names)
    label_dict, label = np.unique(np.array(adata.obs['celltype']), return_inverse=True)
    if is_master:
        with open('label_dict', 'wb') as fp:
            pkl.dump(label_dict, fp)
        with open('label', 'wb') as fp:
            pkl.dump(label, fp)
    label = torch.from_numpy(label)
    data = adata.X

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

    class Identity(nn.Module):
        def __init__(self, dropout=0., h_dim=100, out_dim=10):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 1, (1, 200))
            self.act = nn.ReLU()
            self.fc1 = nn.Linear(SEQ_LEN, 512)
            self.act1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)
            self.fc2 = nn.Linear(512, h_dim)
            self.act2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)
            self.fc3 = nn.Linear(h_dim, out_dim)

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

    ckpt_path = f"/data1/data/corpus/scMODEL/{model_name}_full_model_Zheng68K.pkl"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()

    net = model.module if isinstance(model, DDP) else model

    class_total = defaultdict(int)
    class_correct = defaultdict(int)

    with torch.no_grad():
        for x_v, y_v in val_loader:
            x_v = x_v.to(device)
            y_v = y_v.to(device)
            reps = net(x_v, return_encodings=True)
            logits = net.to_out(reps)
            preds = logits.argmax(dim=1)
            for label_i, pred in zip(y_v.cpu().numpy(), preds.cpu().numpy()):
                class_total[label_i] += 1
                if label_i == pred:
                    class_correct[label_i] += 1

    if is_master:
        print("Per-class Accuracy Report")
        print("Class\tCorrect\tTotal\tAccuracy")
        for idx, class_name in enumerate(label_dict):
            correct = class_correct.get(idx, 0)
            total = class_total.get(idx, 0)
            acc = correct / total if total > 0 else 0.0
            print(f"{class_name}\t{correct}\t{total}\t{round(acc * 100, 2)}%")

    correct_seqs = []
    correct_lbls = []

    with torch.no_grad():
        for x_v, y_v in val_loader:
            x_v = x_v.to(device)
            y_v = y_v.to(device)
            reps = net(x_v, return_encodings=True)
            logits = net.to_out(reps)
            preds = logits.argmax(dim=1)
            correct_mask = preds == y_v
            if correct_mask.any():
                correct_seqs.append(x_v[correct_mask])
                correct_lbls.append(y_v[correct_mask])

    if len(correct_seqs) == 0:
        if is_master:
            print("[WARNING] No correctly predicted samples found.")
        exit()

    seqs = torch.cat(correct_seqs)
    lbls = torch.cat(correct_lbls)

    relevances = []
    iterator = zip(seqs, lbls)
    if is_master:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=len(seqs), desc="Calculating relevance")

    for seq, lbl in iterator:
        net.zero_grad(set_to_none=True)
        seq_input = seq.unsqueeze(0).to(device)

        reps = net(seq_input, return_encodings=True)
        reps.retain_grad()

        logits = net.to_out(reps)
        logits[0, lbl].backward()

        rel = (reps.grad * reps).sum(dim=-1).squeeze().detach().cpu().numpy()
        rel = rel / (np.max(np.abs(rel)) + 1e-12)
        relevances.append(rel)

    relevances = np.stack(relevances)[:, :-1]

    records = []
    for cls in np.unique(lbls.cpu()):
        arr = np.mean(relevances[lbls.cpu() == cls], axis=0)
        top15 = arr.argsort()[-15:][::-1]
        bot15 = arr.argsort()[:15]
        for rank, g in enumerate(top15, 1):
            records.append((label_dict[cls], rank, gene_names[g], arr[g]))
        for rank, g in enumerate(bot15[::-1], 1):
            records.append((label_dict[cls], -rank, gene_names[g], arr[g]))

    df = pd.DataFrame(records, columns=['class', 'rank', 'gene', 'value'])
    if is_master:
        df.to_csv("finetune_gradr_relevance_topbot.csv", index=False)

finally:
    dist.destroy_process_group()
