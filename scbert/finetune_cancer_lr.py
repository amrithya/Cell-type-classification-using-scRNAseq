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

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, default=1136)
parser.add_argument("--epoch", type=int, default=15)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=1e-5)
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
world_size = torch.distributed.get_world_size()

seed_all(SEED + torch.distributed.get_rank())

ckpt_path = f"/data1/data/corpus/scMODEL/{model_name}_full_model_cancer_kidney.pkl"
if os.path.exists(ckpt_path):
    os.remove(ckpt_path)
ckpt_dir = f"/data1/data/corpus/scMODEL/{model_name}_model_cancer_kidney.pkl"

class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0] - 1)
        full_seq = self.data[rand_start].toarray()[0].astype(np.int64)
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        seq_label = int(self.label[rand_start])
        return full_seq, torch.tensor(seq_label, dtype=torch.long, device=device)

    def __len__(self):
        return self.data.shape[0]

print("Loading data...")
data = sc.read_h5ad(args.data_path)
label_dict_cancer, label_cancer = np.unique(np.array(data.obs['class']), return_inverse=True)
with open('label_dict_cancer', 'wb') as fp:
    pkl.dump(label_dict_cancer, fp)
with open('label_cancer', 'wb') as fp:
    pkl.dump(label_cancer, fp)
class_num = np.unique(label_cancer, return_counts=True)[1].tolist()
class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
label = torch.tensor(label_cancer, dtype=torch.long)
data = data.X

acc = []
f1 = []
f1w = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pred_list = pd.Series(['un'] * data.shape[0])

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

if is_master:
    os.makedirs("/data1/data/corpus/scDATA/", exist_ok=True)
    with open("/data1/data/corpus/scDATA/train_loader.pkl", "wb") as f:
        pkl.dump(train_loader, f)
    with open("/data1/data/corpus/scDATA/val_loader.pkl", "wb") as f:
        pkl.dump(val_loader, f)

model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    local_attn_heads=0,
    g2v_position_emb=POS_EMBED_USING
)

path = args.model_path
ckpt = torch.load(path, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])

for param in model.parameters():
    param.requires_grad = False
    param.data = param.data.float()
model.to_out = nn.Linear(model.dim, label_dict_cancer.shape[0])
model.to_out.requires_grad_(True)

model = model.to(device).to(dtype=torch.float32)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=15,
    cycle_mult=2,
    max_lr=LEARNING_RATE,
    min_lr=1e-6,
    warmup_steps=5,
    gamma=0.9
)
loss_fn = nn.CrossEntropyLoss(weight=None).to(local_rank)

dist.barrier()
trigger_times = 0
max_acc = 0.0
start_epoch = 1

if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'epoch' in checkpoint and checkpoint['epoch'] < EPOCHS:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            from collections import OrderedDict
            state_dict = checkpoint['model_state_dict']
            new_state_dict = OrderedDict()
            if next(iter(state_dict)).startswith("module."):
                for k, v in state_dict.items():
                    new_state_dict[k[len("module."):]] = v
            else:
                for k, v in state_dict.items():
                    new_state_dict["module." + k] = v
            model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

for i in range(start_epoch, EPOCHS):
    train_loader.sampler.set_epoch(i)
    model.train()
    dist.barrier()
    running_loss = 0.0
    cum_acc = 0.0
    for index, (data, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {i} - Training")):
        index += 1
        data, labels = data.to(device), labels.to(device)
        if index % GRADIENT_ACCUMULATION != 0:
            with model.no_sync():
                logits = model(data).float()
                loss = loss_fn(logits, labels)
                loss.backward()
        else:
            logits = model(data).float()
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item()
        softmax = nn.Softmax(dim=-1)
        final = softmax(logits).argmax(dim=-1)
        pred_num = labels.size(0)
        correct_num = torch.eq(final, labels).sum(dim=-1)
        cum_acc += torch.true_divide(correct_num, pred_num).mean().item()

    epoch_loss = get_reduced(running_loss / index, local_rank, 0, world_size)
    epoch_acc = get_reduced(100 * cum_acc / index, local_rank, 0, world_size)
    if is_master:
        print(f'==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
    dist.barrier()
    scheduler.step()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        dist.barrier()
        running_loss = 0.0
        predictions = []
        truths = []
        with torch.no_grad():
            for index, (data_v, labels_v) in enumerate(tqdm(val_loader, desc=f"Epoch {i} - Validation")):
                data_v, labels_v = data_v.to(device), labels_v.to(device)
                logits = model(data_v)
                loss = loss_fn(logits, labels_v)
                running_loss += loss.item()
                softmax = nn.Softmax(dim=-1)
                final_prob = softmax(logits)
                final = final_prob.argmax(dim=-1)
                final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
                predictions.append(final)
                truths.append(labels_v)

        predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
        truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)
        no_drop = predictions != -1
        predictions = np.array((predictions[no_drop]).cpu())
        truths = np.array((truths[no_drop]).cpu())
        cur_acc = accuracy_score(truths, predictions)
        f1 = f1_score(truths, predictions, average='macro')
        val_loss = get_reduced(running_loss / index, local_rank, 0, world_size)
        if is_master:
            print(f'==  Epoch: {i} | Validation Loss: {val_loss:.6f} | F1 Score: {f1:.6f}  ==')
            print(confusion_matrix(truths, predictions))
            print(classification_report(truths, predictions, target_names=label_dict_cancer.tolist(), digits=4))
        if cur_acc > max_acc:
            max_acc = cur_acc
            trigger_times = 0
            save_ckpt_cancer(i, model, optimizer, scheduler, val_loss, model_name)
        else:
            trigger_times += 1
            if trigger_times > PATIENCE:
                break
        del predictions, truths
