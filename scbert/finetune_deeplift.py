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
from collections import Counter, defaultdict

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
is_master = rank == 0

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
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, (1, 200))
        self.act = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(in_features=SEQ_LEN, out_features=512)
        self.act1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(512, h_dim)
        self.act2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(dropout)
        self.fc3 = torch.nn.Linear(h_dim, out_dim)

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
adata = sc.read_h5ad(args.data_path)
gene_names = np.array(adata.var_names)

label_dict, label = np.unique(np.array(adata.obs['celltype']), return_inverse=True)
label = torch.from_numpy(label)
data = adata.X

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
for _, index_val in sss.split(data, label):
    data_val, label_val = data[index_val], label[index_val]

val_label_counts = Counter(label_val.numpy())

val_dataset = SCDataset(data_val, label_val)
val_sampler = DistributedSampler(val_dataset, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

if is_master:
    print(f"Validation dataset size: {len(val_dataset)}")
    for class_idx in range(len(label_dict)):
        print(f"{label_dict[class_idx]}: {val_label_counts[class_idx]}")

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
print("Checkpoint loaded successfully.")

class ModelFromEmbeddings(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, embedded_input):
        return self.model.to_out(embedded_input)

wrapper_model = ModelFromEmbeddings(model.module)
dl = DeepLift(wrapper_model)

all_attrs = []
all_labels = []
correct_counts = Counter()
total_counts = Counter()

for data_v, labels_v in tqdm(val_loader, desc="DeepLIFT on val"):
    data_v = data_v.to(device)
    labels_v = labels_v.to(device)
    with torch.no_grad():
        embedded_input = model.module.token_emb(data_v)
        outputs = model.module.to_out(embedded_input)
        preds = outputs.argmax(dim=1)

    for class_idx in labels_v.cpu().numpy():
        total_counts[class_idx] += 1

    correct_mask = (preds == labels_v)
    if correct_mask.sum() == 0:
        continue

    data_correct = data_v[correct_mask]
    labels_correct = labels_v[correct_mask]

    batch_counts = Counter(labels_correct.cpu().numpy())
    for class_idx, count in batch_counts.items():
        correct_counts[class_idx] += count

    embedded_input_correct = model.module.token_emb(data_correct)
    baseline = torch.zeros_like(embedded_input_correct)

    embedded_input_correct = embedded_input_correct.detach().clone().requires_grad_(True)
    attributions = dl.attribute(inputs=embedded_input_correct, baselines=baseline, target=labels_correct)

    all_attrs.append(attributions.detach().cpu().numpy())
    all_labels.append(labels_correct.detach().cpu().numpy())

all_attrs_tensor = torch.tensor(np.concatenate(all_attrs, axis=0)).to(device)
all_labels_tensor = torch.tensor(np.concatenate(all_labels, axis=0)).to(device)

attrs_gather = [torch.zeros_like(all_attrs_tensor) for _ in range(dist.get_world_size())]
labels_gather = [torch.zeros_like(all_labels_tensor) for _ in range(dist.get_world_size())]

dist.all_gather(attrs_gather, all_attrs_tensor)
dist.all_gather(labels_gather, all_labels_tensor)


dist.barrier()

if is_master:
    all_attrs = torch.cat(attrs_gather, dim=0).numpy()
    all_labels = torch.cat(labels_gather, dim=0).numpy()

    infer_label_counts = Counter(all_labels)
    val_total_sum = sum(val_label_counts.values())
    infer_total_sum = sum(infer_label_counts.values())
    print(f"Validation sample total = {val_total_sum} | Processed total = {infer_total_sum}")
    if val_total_sum != infer_total_sum:
        print("WARNING: Number of processed samples does not match validation dataset!")

    if all_attrs.ndim > 2:
        all_attrs = all_attrs.mean(axis=tuple(range(2, all_attrs.ndim)))

    gene_attrs = all_attrs[:, :-1]
    results = []

    for class_idx in np.unique(all_labels):
        class_mask = all_labels == class_idx
        class_attrs = gene_attrs[class_mask]
        mean_attrs = class_attrs.mean(axis=0)

        top15_idx = np.argsort(mean_attrs)[-15:][::-1]
        bottom15_idx = np.argsort(mean_attrs)[:15]

        for rank, i in enumerate(top15_idx):
            results.append({
                'celltype': label_dict[class_idx],
                'rank': rank + 1,
                'gene': gene_names[i],
                'attribution': mean_attrs[i],
                'type': 'top'
            })

        for rank, i in enumerate(bottom15_idx):
            results.append({
                'celltype': label_dict[class_idx],
                'rank': rank + 1,
                'gene': gene_names[i],
                'attribution': mean_attrs[i],
                'type': 'bottom'
            })

    os.makedirs(args.output_dir, exist_ok=True)
    df_all = pd.DataFrame(results)
    df_all.to_csv(os.path.join(args.output_dir, 'correct_deeplift_top_bottom15_genes.csv'), index=False)
    print(f"Saved results to {args.output_dir}correct_deeplift_top_bottom15_genes.csv")

dist.destroy_process_group()
