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
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

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

class PerformerLMWithAttn(PerformerLM):
    def forward(self, x, return_attn=False, return_encodings=False):
        out = super().forward(x)
        last_attn = None
        if hasattr(self, 'performer') and hasattr(self.performer, 'layers'):
            last_layer = self.performer.layers[-1]
            if hasattr(last_layer, 'attn') and hasattr(last_layer.attn, 'last_attn'):
                last_attn = last_layer.attn.last_attn
        if return_encodings:
            return out, last_attn
        if return_attn:
            return last_attn
        return out

try:
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
    except Exception as e:
        if is_master:
            print(f"[ERROR] Failed to load data: {e}")
        raise

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    for train_idx, val_idx in sss.split(data, label):
        train_dataset = SCDataset(data[train_idx], label[train_idx])
        val_dataset = SCDataset(data[val_idx], label[val_idx])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

    model = PerformerLMWithAttn(
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
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt['model_state_dict']
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('to_out.')}
        model.load_state_dict(filtered_state_dict, strict=False)
    except Exception as e:
        if is_master:
            print(f"[ERROR] Failed to load checkpoint: {e}")
        raise

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()

    net = model.module if isinstance(model, DDP) else model

    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None
            self.hook_layers()

        def hook_layers(self):
            def forward_hook(module, input, output):
                self.activations = output.detach()
            def backward_hook(module, grad_in, grad_out):
                self.gradients = grad_out[0].detach()

            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_backward_hook(backward_hook)

        def __call__(self, x, class_idx):
            self.model.zero_grad()
            x = x.requires_grad_(True)
            outputs = self.model(x)
            one_hot = torch.zeros_like(outputs)
            if isinstance(class_idx, torch.Tensor):
                class_idx = class_idx.cpu().numpy()
            if isinstance(class_idx, (list, np.ndarray)) and len(class_idx) > 1:
                losses = []
                for i, c in enumerate(class_idx):
                    losses.append(outputs[i, c])
                loss = torch.stack(losses).sum()
            else:
                loss = outputs[0, class_idx] if not isinstance(class_idx, (list,np.ndarray)) else outputs[0,class_idx[0]]
            loss.backward(retain_graph=True)
            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
            activations = self.activations[0]  
            for i in range(activations.shape[0]):
                activations[i, ...] *= pooled_gradients[i]
            heatmap = torch.mean(activations, dim=0).cpu().numpy()
            heatmap = np.maximum(heatmap, 0)
            heatmap = heatmap / np.max(heatmap) if np.max(heatmap) != 0 else heatmap
            heatmap = heatmap.flatten()
            return heatmap

    gradcam = GradCAM(model.module, model.module.to_out.conv1)

    records = []

    model.eval()
    with torch.no_grad():
        class_gene_scores = {c: [] for c in range(CLASS)}

        for data, labels in tqdm(val_loader, desc="Validation Batches"):
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            preds = outputs.argmax(dim=1)

            correct_mask = preds == labels
            if correct_mask.sum() == 0:
                continue

            correct_data = data[correct_mask]
            correct_labels = labels[correct_mask]
            correct_preds = preds[correct_mask]

            cams = gradcam(correct_data, class_idx=correct_preds)
            if cams.ndim == 2:
                pass
            elif cams.ndim == 1:
                cams = cams[None, :]  

            for c in torch.unique(correct_preds):
                c = c.item()
                idxs = (correct_preds == c).nonzero(as_tuple=True)[0].cpu().numpy()
                class_c_scores = cams[idxs]
                class_gene_scores[c].append(class_c_scores)

        for c in tqdm(class_gene_scores, desc="Computing class gene scores"):
            if len(class_gene_scores[c]) == 0:
                continue
            scores = np.concatenate(class_gene_scores[c], axis=0)
            mean_scores = scores.mean(axis=0)

            top15_idx = np.argsort(mean_scores)[-15:][::-1]
            bottom15_idx = np.argsort(mean_scores)[:15]

            for rank, gene_idx in enumerate(top15_idx, 1):
                records.append([label_dict[c], rank, gene_names[gene_idx], mean_scores[gene_idx]])

            for rank, gene_idx in enumerate(bottom15_idx, 16):
                records.append([label_dict[c], rank, gene_names[gene_idx], mean_scores[gene_idx]])

    df = pd.DataFrame(records, columns=['class', 'rank', 'gene', 'value'])
    df.to_csv("gradcam_class_top_bottom_genes.csv", index=False)
    if is_master:
        print("Saved class-wise top and bottom genes to class_top_bottom_genes.csv")

except Exception as e:
    if is_master:
        print(f"[ERROR] Unexpected error: {e}")
    raise
