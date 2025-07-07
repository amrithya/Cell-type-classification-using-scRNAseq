# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import scanpy as sc
import torch
from performer_pytorch import PerformerLM
from tqdm import tqdm

parser = argparse.ArgumentParser() 
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, default=16906)
parser.add_argument("--data_path", type=str, default='./data/data.h5ad')
parser.add_argument("--model_path", type=str, default='./model.pth')
parser.add_argument("--save_dir", type=str, default='./attention/')
args = parser.parse_args()

SEQ_LEN = args.gene_num + 1
CLASS = args.bin_num + 2

data_dir = args.data_path
model_dir = args.model_path
save_dir = args.save_dir

device = torch.device("cuda")
print('            =======  Config over  ======= \n')

data = sc.read_h5ad(data_dir)

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
        num_tokens = CLASS,
        dim = 200,
        depth = 6,
        max_seq_len = SEQ_LEN,
        heads = 10,
        local_attn_heads = 0,
        g2v_position_emb = False
    )
    print(f'            =======  Model defined  ======= \n')

    ckpt_path = model_dir
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt['model_state_dict']
        ignored_keys = [k for k in state_dict.keys() if k.startswith("to_out.") or k.startswith("pos_emb.")]
        for k in ignored_keys:
            del state_dict[k]

        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        raise

    model = model.to(device)
    print('            =======  Predict start  ======= \n')

    batch_size = data_alpha.shape[0]
    model.eval()
    with torch.no_grad():
        final_mtx = torch.zeros(batch_size, data_alpha.shape[1]+1).to(device)
        for index in tqdm(range(batch_size), desc=f"Processing {cellind}"):
            full_seq = data_alpha[index].toarray()[0]
            full_seq[full_seq > (CLASS - 2)] = CLASS - 2
            full_seq = torch.from_numpy(full_seq).long()
            full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
            full_seq = full_seq.unsqueeze(0)
            _, attn_map = model(full_seq, output_attentions=True)
            attn_map = attn_map.mean((0,1,2))
            attn_map /= attn_map.sum()
            final_mtx[index] = attn_map

    final_mtx = final_mtx.detach().cpu().numpy()
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'full_attn_sum_{cellind}.npy'), final_mtx)
    print(f'            =======  Predict end  ======= \n')
