import os
import gc
import argparse
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from performer_pytorch import PerformerLM
import scanpy as sc
import pickle as pkl
from tqdm import tqdm
import shap

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
parser.add_argument("--data_path", type=str, default='/data1/data/corpus/scDATA/Zheng68K.h5ad')
parser.add_argument("--model_path", type=str, default='/data1/data/corpus/scMODEL/finetune_full_model_Zheng68K.pkl')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/')
parser.add_argument("--model_name", type=str, default='finetune')
args = parser.parse_args()

rank = int(os.environ["RANK"])
local_rank = args.local_rank
is_master = local_rank == 0

dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every
CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed
model_name = args.model_name

random.seed(SEED + rank)
torch.manual_seed(SEED + rank)
torch.cuda.manual_seed_all(SEED + rank)

class SCDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        seq_label = self.label[rand_start]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]

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

def load_data(path):
    data = sc.read_h5ad(path)
    label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)
    with open('label_dict', 'wb') as f:
        pkl.dump(label_dict, f)
    with open('label', 'wb') as f:
        pkl.dump(label, f)
    return data.X, torch.from_numpy(label), label_dict

DATA_PATH = args.data_path
MODEL_PATH = args.model_path
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

data, label, label_dict = load_data(DATA_PATH)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2021)
index_train, index_val = next(sss.split(data, label))
data_val, label_val = data[index_val], label[index_val]

val_dataset = SCDataset(data_val, label_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

if os.path.exists(f"{CACHE_DIR}/correct_inputs.npy") and os.path.exists(f"{CACHE_DIR}/correct_labels.npy"):
    correct_inputs_np = np.load(f"{CACHE_DIR}/correct_inputs.npy")
    correct_labels = torch.from_numpy(np.load(f"{CACHE_DIR}/correct_labels.npy"))
else:
    model = PerformerLM(num_tokens=CLASS, dim=200, depth=6, max_seq_len=SEQ_LEN, heads=10, local_attn_heads=0, g2v_position_emb=True)
    model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
    model = model.to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_inputs, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validation Batches"):
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu()
            all_inputs.append(x.cpu())
            all_labels.append(y)
            all_preds.append(preds)

    all_inputs = torch.cat(all_inputs)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    correct_mask = all_preds == all_labels
    correct_inputs = all_inputs[correct_mask]
    correct_labels = all_labels[correct_mask]
    correct_inputs_np = correct_inputs.numpy()

    np.save(f"{CACHE_DIR}/correct_inputs.npy", correct_inputs_np)
    np.save(f"{CACHE_DIR}/correct_labels.npy", correct_labels.numpy())

if dist.is_initialized():
    dist.destroy_process_group()

def model_predict(x):
    x_tensor = torch.from_numpy(x).long()
    model = PerformerLM(num_tokens=CLASS, dim=200, depth=6, max_seq_len=SEQ_LEN, heads=10, local_attn_heads=0, g2v_position_emb=True)
    model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs.cpu().numpy()

if is_master:
    target_idx = 0
    target_sample = correct_inputs_np[target_idx:target_idx+1]
    background = correct_inputs_np[np.random.choice(correct_inputs_np.shape[0], 10, replace=False)]

    explainer = shap.KernelExplainer(model_predict, background)
    shap_vals = explainer.shap_values(target_sample, nsamples=50)

    gene_names = [f"gene_{i}" for i in range(SEQ_LEN - 1)]
    shap_val_target = shap_vals[np.argmax(model_predict(target_sample))]
    top_15_idx = np.argsort(shap_val_target)[-15:][::-1]
    bottom_15_idx = np.argsort(shap_val_target)[:15]

    top_genes = [(gene_names[i], shap_val_target[i]) for i in top_15_idx]
    bottom_genes = [(gene_names[i], shap_val_target[i]) for i in bottom_15_idx]

    df_top = pd.DataFrame(top_genes, columns=["gene", "shap_value"])
    df_bottom = pd.DataFrame(bottom_genes, columns=["gene", "shap_value"])
    df_top["rank"] = "top"
    df_bottom["rank"] = "bottom"
    df_shap = pd.concat([df_top, df_bottom])

    os.makedirs("results", exist_ok=True)
    df_shap.to_csv("results/shap_single_sample.csv", index=False)

    del shap_vals, background, target_sample
    gc.collect()
