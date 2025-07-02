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

if args.local_rank != -1:
    rank = int(os.environ["RANK"])
    local_rank = args.local_rank
    is_master = local_rank == 0
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_master = True

SEED = args.seed
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

class SCDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (args.bin_num - 2)] = args.bin_num - 2
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
        self.fc1 = nn.Linear(args.gene_num + 1, 512)
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
    with open('label_dict.pkl', 'wb') as f:
        pkl.dump(label_dict, f)
    with open('label.pkl', 'wb') as f:
        pkl.dump(label, f)
    return data.X, torch.from_numpy(label), label_dict

if __name__ == "__main__":
    data, label, label_dict = load_data(args.data_path)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2021)
    index_train, index_val = next(sss.split(data, label))
    data_val, label_val = data[index_val], label[index_val]
    val_dataset = SCDataset(data_val, label_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    CACHE_DIR = "cache"
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(f"{CACHE_DIR}/correct_inputs.npy") and os.path.exists(f"{CACHE_DIR}/correct_labels.npy"):
        correct_inputs_np = np.load(f"{CACHE_DIR}/correct_inputs.npy")
        correct_labels = torch.from_numpy(np.load(f"{CACHE_DIR}/correct_labels.npy"))
    else:
        model = PerformerLM(
            num_tokens=args.bin_num + 2,
            dim=200,
            depth=6,
            max_seq_len=args.gene_num + 1,
            heads=10,
            local_attn_heads=0,
            g2v_position_emb=args.pos_embed
        )
        model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
        model = model.to(device)
        if args.local_rank != -1:
            model = DDP(model, device_ids=[local_rank])
        checkpoint = torch.load(args.model_path, map_location=device)
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

    if args.local_rank != -1:
        dist.destroy_process_group()

    if is_master:
        model = PerformerLM(
            num_tokens=args.bin_num + 2,
            dim=200,
            depth=6,
            max_seq_len=args.gene_num + 1,
            heads=10,
            local_attn_heads=0,
            g2v_position_emb=args.pos_embed
        )
        model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
        model = model.to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        target_idx = 0
        target_sample = correct_inputs_np[target_idx:target_idx+1]
        background_samples = min(50, correct_inputs_np.shape[0])
        background = correct_inputs_np[np.random.choice(correct_inputs_np.shape[0], background_samples, replace=False)]
        background_tensor = torch.from_numpy(background).long().to(device)
        target_tensor = torch.from_numpy(target_sample).long().to(device)
        torch.cuda.empty_cache()
        gc.collect()
        try:
            print("Starting SHAP analysis with DeepExplainer...")
            explainer = shap.DeepExplainer(model, background_tensor)
            batch_size = 5
            shap_values = []
            for i in range(0, len(target_tensor), batch_size):
                batch = target_tensor[i:i+batch_size]
                shap_values_batch = explainer.shap_values(batch)
                shap_values.append(shap_values_batch[0])
                torch.cuda.empty_cache()
            shap_values = np.concatenate(shap_values)
            gene_names = [f"gene_{i}" for i in range(args.gene_num)]
            for i in range(len(target_tensor)):
                shap_val_target = shap_values[i]
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
                df_shap.to_csv(f"results/shap_sample_{i}.csv", index=False)
                print(f"Saved SHAP results for sample {i}")
        except Exception as e:
            print(f"SHAP analysis failed: {str(e)}")
            with open("shap_error.log", "w") as f:
                f.write(str(e))
        print("SHAP analysis completed.")
