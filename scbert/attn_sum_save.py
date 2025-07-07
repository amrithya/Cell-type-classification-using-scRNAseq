# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader
from performer_pytorch import PerformerLM
from tqdm import tqdm

parser = argparse.ArgumentParser() 
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, default=16906)
parser.add_argument("--data_path", type=str, default='./data/data.h5ad')
parser.add_argument("--model_path", type=str, default='./model.pth')
parser.add_argument("--save_dir", type=str, default='./attention/')
parser.add_argument("--batch_size", type=int, default=2)
args = parser.parse_args()

SEQ_LEN = args.gene_num + 1
CLASS = args.bin_num + 2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use only GPU 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('            =======  Config over  ======= \n')

data = sc.read_h5ad(args.data_path)
index_labels = data.obs['celltype']
cellinds = list(set(index_labels.tolist()))
label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)
data_counts = data.X

for cellind in cellinds:
    print(cellind)
    mask = (index_labels == cellind).to_numpy()
    indices = np.where(mask)[0]
    data_alpha = data_counts[indices]

    model = PerformerLM(
        num_tokens=CLASS,
        dim=200,
        depth=6,
        max_seq_len=SEQ_LEN,
        heads=10,
        local_attn_heads=0,
        g2v_position_emb=False
    )
    print(f'            =======  Model defined  ======= \n')

    try:
        ckpt = torch.load(args.model_path, map_location=device)
        state_dict = ckpt['model_state_dict']
        ignored_keys = [k for k in state_dict.keys() if k.startswith("to_out.") or k.startswith("pos_emb.")]
        for k in ignored_keys:
            del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        raise

    model = model.to(device)
    model.eval()

    batch_size = args.batch_size
    input_seqs = []

    for i in range(data_alpha.shape[0]):
        full_seq = data_alpha[i].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0])))
        input_seqs.append(full_seq)

    input_seqs = torch.stack(input_seqs)
    loader = DataLoader(input_seqs, batch_size=batch_size, shuffle=False)

    all_attn = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Processing {cellind}"):
            batch = batch.to(device)
            _, attn_map = model(batch, output_attentions=True)
            attn_map = attn_map.mean((1, 2, 3))
            print("attn_map shape:", attn_map.shape)
            if attn_map.dim() == 1:
                attn_map /= attn_map.sum(dim=0, keepdim=True)
            else:
                attn_map /= attn_map.sum(dim=1, keepdim=True)
            all_attn.append(attn_map)

    final_mtx = torch.cat(all_attn, dim=0).detach().cpu().numpy()
    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, f'full_attn_sum_{cellind}.npy'), final_mtx)
    print(f'            =======  Predict end  ======= \n')
