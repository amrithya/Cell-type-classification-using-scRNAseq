import os
import random
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from collections import defaultdict
from performer_pytorch import PerformerLM
from captum.attr import LayerIntegratedGradients
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, default=16906)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--pos_embed", type=bool, default=True)
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad')
parser.add_argument("--model_path", type=str, default='./panglao_pretrained.pth')
parser.add_argument("--output_dir", type=str, default='./IG_outputs/')
args = parser.parse_args()

rank = int(os.environ["RANK"])
local_rank = args.local_rank
is_master = local_rank == 0

SEED = args.seed
BATCH_SIZE = args.batch_size
SEQ_LEN = args.gene_num + 1
CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed

dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

random.seed(SEED + rank)
torch.manual_seed(SEED + rank)
torch.cuda.manual_seed_all(SEED + rank)

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

class Identity(torch.nn.Module):
    def __init__(self, dropout=0., h_dim=100, out_dim=10):
        super(Identity, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, (1, 200))
        self.act = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(dropout)
        self.fc3 = torch.nn.Linear(in_features=h_dim, out_features=11, bias=True)

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

data = sc.read_h5ad(args.data_path)
label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)
label = torch.from_numpy(label)
data_matrix = data.X

print(f"[RANK {rank}] Loaded data: {data_matrix.shape}, Labels: {label.shape}")

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
for _, index_val in sss.split(data_matrix, label):
    data_val, label_val = data_matrix[index_val], label[index_val]

print(f"[RANK {rank}] Validation split: {data_val.shape}, Labels: {label_val.shape}")

val_dataset = SCDataset(data_val, label_val)
val_sampler = DistributedSampler(val_dataset, shuffle=False)
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
model.to_out = Identity(dropout=0., h_dim=128, out_dim=11)

ckpt_path = args.model_path
ckpt = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])

print(f"[RANK {rank}] Loaded model checkpoint from: {ckpt_path}")

model = model.float()
model.to(device)
model.eval()

print(f"[RANK {rank}] Model moved to device: {device} and set to eval mode.")

def model_forward(input_ids):
    input_ids = input_ids.long().to(device)
    return model(input_ids)

embedding_layer = model.token_emb
lig = LayerIntegratedGradients(model_forward, embedding_layer)

def construct_input_and_baseline(seq):
    pad_token = 0
    seq = seq.to(dtype=torch.long)
    baseline = torch.full_like(seq, pad_token).to(dtype=torch.long)
    return seq.unsqueeze(0).to(device), baseline.unsqueeze(0).to(device)

val_sampler.set_epoch(0)
print(f"[RANK {rank}] Starting attribution loop on {len(val_loader)} batches...")

attr_accumulator = defaultdict(list)

val_iter = val_loader
if is_master:
    val_iter = tqdm(val_loader, desc=f"[RANK {rank}] IG Attribution")

for batch_idx, (seqs, labels) in enumerate(val_iter):
    with torch.no_grad():
        outputs = model(seqs.to(device))
        preds = torch.argmax(outputs, dim=1)
    for i in range(seqs.size(0)):
        if preds[i] == labels[i].to(device):
            input_ids, baseline_ids = construct_input_and_baseline(seqs[i])
            torch.cuda.empty_cache()
            attr, delta = lig.attribute(inputs=input_ids,
                                        baselines=baseline_ids,
                                        target=labels[i].item(),
                                        n_steps=5,
                                        return_convergence_delta=True)
            attr_sum = attr.sum(dim=-1).squeeze(0)
            attr_accumulator[labels[i].item()].append(attr_sum.detach().cpu())

if is_master:
    os.makedirs(args.output_dir, exist_ok=True)
    records = []
    for cls_idx, attr_list in attr_accumulator.items():
        cls_attr = torch.stack(attr_list).mean(dim=0)
        topk = torch.topk(cls_attr, 15)
        bottomk = torch.topk(-cls_attr, 15)
        class_name = label_dict[cls_idx]
        for rank, gene_idx in enumerate(topk.indices.tolist()):
            gene_name = data.var_names[gene_idx]
            records.append({
                "class": class_name,
                "rank": rank + 1,
                "type": "top",
                "gene_name": gene_name,
                "score": topk.values[rank].item()
            })
        for rank, gene_idx in enumerate(bottomk.indices.tolist()):
            gene_name = data.var_names[gene_idx]
            records.append({
                "class": class_name,
                "rank": rank + 1,
                "type": "bottom",
                "gene_name": gene_name,
                "score": -bottomk.values[rank].item()
            })
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.output_dir, 'IG_top_bottom_genes_per_class.csv'), index=False)

print(f"[RANK {rank}] Attribution done. Destroying process group.")
if dist.is_initialized():
    dist.destroy_process_group()
