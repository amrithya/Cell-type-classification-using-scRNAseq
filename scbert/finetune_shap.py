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
import shap

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

class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        full_seq = self.data[index].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0])))
        seq_label = self.label[index]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]

def load_data(data_path):
    data = sc.read_h5ad(data_path)
    label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)
    with open('label_dict', 'wb') as fp:
        pkl.dump(label_dict, fp)
    with open('label', 'wb') as fp:
        pkl.dump(label, fp)
    return data.X, torch.from_numpy(label), label_dict

def get_unwrapped_model(model_ddp):
    return model_ddp.module if hasattr(model_ddp, 'module') else model_ddp

CLASS = 7 
SEQ_LEN = 16907
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4
MODEL_PATH = '/data1/data/corpus/scMODEL/finetune_full_model_Zheng68K.pkl'
DATA_PATH = '/data1/data/corpus/scDATA/Zheng68K.h5ad'

data, label, label_dict = load_data(DATA_PATH)

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2021)
index_train, index_val = next(sss.split(data, label))
data_val, label_val = data[index_val], label[index_val]

val_dataset = SCDataset(data_val, label_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    local_attn_heads=0,
    g2v_position_emb=True
)

model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
model = model.to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()


all_inputs = []
all_labels = []
all_preds = []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(DEVICE)
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

correct_inputs_np = correct_inputs[:, :-1].numpy()  # exclude CLS token

def model_predict(x):
    x_tensor = torch.tensor(x, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
    return probs

background_size = min(50, correct_inputs_np.shape[0])
background = correct_inputs_np[np.random.choice(correct_inputs_np.shape[0], background_size, replace=False)]

explainer = shap.KernelExplainer(model_predict, background)

gene_names = [f"gene_{i}" for i in range(SEQ_LEN - 1)]

top_bottom_genes = []

for class_idx in range(CLASS):
    class_indices = (correct_labels.numpy() == class_idx)
    if not np.any(class_indices):
        continue
    class_inputs = correct_inputs_np[class_indices]

    shap_vals_accum = []
    batch_size = 10
    for j in range(0, class_inputs.shape[0], batch_size):
        batch = class_inputs[j:j+batch_size]
        try:
            shap_vals_batch = explainer.shap_values(batch, nsamples=100)
            shap_vals_accum.append(shap_vals_batch[class_idx])
        except Exception as e:
            print(f"SHAP batch error: {e}")
            continue

    if len(shap_vals_accum) == 0:
        continue

    class_shap_vals = np.vstack(shap_vals_accum)
    mean_shap = np.mean(class_shap_vals, axis=0)

    top_15_idx = np.argsort(mean_shap)[-15:][::-1]
    bottom_15_idx = np.argsort(mean_shap)[:15]

    top_genes = [(gene_names[i], mean_shap[i]) for i in top_15_idx]
    bottom_genes = [(gene_names[i], mean_shap[i]) for i in bottom_15_idx]

    for gene, val in top_genes:
        top_bottom_genes.append([class_idx, 'top', gene, val])
    for gene, val in bottom_genes:
        top_bottom_genes.append([class_idx, 'bottom', gene, val])

df_shap = pd.DataFrame(top_bottom_genes, columns=['class', 'rank', 'gene', 'mean_shap'])
df_shap.to_csv(f"results/top_bottom15_genes_shap_only.csv", index=False)

print("SHAP analysis completed and saved.")
