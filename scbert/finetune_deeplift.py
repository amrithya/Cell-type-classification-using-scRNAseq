import os
import random
import argparse
import pickle as pkl
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from performer_pytorch import PerformerLM
from captum.attr import DeepLift
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, default=16906)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--pos_embed", type=bool, default=True)
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad')
parser.add_argument("--model_path", type=str, default='./panglao_pretrained.pth')
parser.add_argument("--output_dir", type=str, default='./deeplift_outputs/')
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
        self.fc3 = torch.nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

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

print("Loading data...")
data = sc.read_h5ad(args.data_path)
label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)
label = torch.from_numpy(label)
data = data.X

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
for _, index_val in sss.split(data, label):
    data_val, label_val = data[index_val], label[index_val]

val_dataset = SCDataset(data_val, label_val)
val_sampler = DistributedSampler(val_dataset, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

print("Building model...")
model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    local_attn_heads=0,
    g2v_position_emb=POS_EMBED_USING
)
model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
model = model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

ckpt_path = "/data1/data/corpus/scMODEL/finetune_full_model_Zheng68K.pkl"

print("Loading finetuned checkpoint...")
ckpt = torch.load(ckpt_path, map_location='cpu')
model.module.load_state_dict(ckpt['model_state_dict'])

model.eval()

dl = DeepLift(model.module)

os.makedirs(args.output_dir, exist_ok=True)

if is_master:
    print(f"Running DeepLIFT attribution on full validation set with {len(val_loader)} batches")

all_attrs = []
all_labels = []

with torch.no_grad():
    for batch_idx, (data_v, labels_v) in enumerate(tqdm(val_loader, desc="DeepLIFT on val")):
        data_v = data_v.to(device)
        labels_v = labels_v.to(device)
        baseline = torch.zeros_like(data_v).to(device)
        attributions = dl.attribute(data_v, baselines=baseline, target=labels_v)

        if is_master:
            all_attrs.append(attributions.cpu().numpy())
            all_labels.append(labels_v.cpu().numpy())

dist.barrier()

if is_master:
    all_attrs = np.concatenate(all_attrs, axis=0)  # shape (N_samples, seq_len, ...)
    all_labels = np.concatenate(all_labels, axis=0)  # shape (N_samples,)

    if all_attrs.ndim > 2:
        all_attrs = all_attrs.mean(axis=tuple(range(2, all_attrs.ndim)))

    gene_attrs = all_attrs[:, :-1]  # exclude appended token

    results = {}
    for class_idx in np.unique(all_labels):
        class_mask = all_labels == class_idx
        class_attrs = gene_attrs[class_mask]
        mean_attrs = class_attrs.mean(axis=0)

        top15_idx = np.argsort(mean_attrs)[-15:][::-1]
        bottom15_idx = np.argsort(mean_attrs)[:15]

        results[class_idx] = {
            'top15_genes': top15_idx,
            'top15_values': mean_attrs[top15_idx],
            'bottom15_genes': bottom15_idx,
            'bottom15_values': mean_attrs[bottom15_idx]
        }

    gene_names = np.array(data.var_names) if hasattr(data, 'var_names') else np.array([f'gene_{i}' for i in range(SEQ_LEN - 1)])

    for class_idx, res in results.items():
        class_name = label_dict[class_idx]
        df_top = pd.DataFrame({
            'gene': gene_names[res['top15_genes']],
            'attribution': res['top15_values']
        })
        df_bottom = pd.DataFrame({
            'gene': gene_names[res['bottom15_genes']],
            'attribution': res['bottom15_values']
        })
        df_top.to_csv(os.path.join(args.output_dir, f'{class_name}_top15_genes.csv'), index=False)
        df_bottom.to_csv(os.path.join(args.output_dir, f'{class_name}_bottom15_genes.csv'), index=False)

dist.destroy_process_group()
