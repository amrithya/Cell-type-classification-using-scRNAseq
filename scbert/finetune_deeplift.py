import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader, DistributedSampler
from performer_pytorch import PerformerLM
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from captum.attr import DeepLift

parser = argparse.ArgumentParser()
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, default=16906)
parser.add_argument("--data_path", type=str, default='./data/data.h5ad')
parser.add_argument("--model_path", type=str, default='./model.pth')
parser.add_argument("--save_dir", type=str, default='./deeplift/')
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

SEQ_LEN = args.gene_num + 1
CLASS = args.bin_num + 2
BATCH_SIZE = args.batch_size
SEED = args.seed

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = sc.read_h5ad(args.data_path)
index_labels = data.obs['celltype']
label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)
data_counts = data.X
gene_names = data.var_names

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
for _, index_val in sss.split(data_counts, label):
    data_val, label_val = data_counts[index_val], label[index_val]

class SCDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        full_seq = self.data[idx].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0])))
        label = self.labels[idx]
        return full_seq, label

val_dataset = SCDataset(data_val, label_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=False)

model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    local_attn_heads=0,
    g2v_position_emb=False
)
ckpt = torch.load(args.model_path, map_location=device)
state_dict = ckpt['model_state_dict']
ignored_keys = [k for k in state_dict.keys() if k.startswith("to_out.") or k.startswith("pos_emb.")]
for k in ignored_keys:
    del state_dict[k]
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

deeplift = DeepLift(model)

all_relevances = []
all_labels = []

for inputs, labels_batch in tqdm(val_loader):
    inputs = inputs.to(device)
    labels_batch = labels_batch.to(device)

    baseline = torch.zeros_like(inputs).to(device)

    batch_relevances = []
    for i in range(inputs.size(0)):
        input_i = inputs[i].unsqueeze(0)
        baseline_i = baseline[i].unsqueeze(0)
        target_i = labels_batch[i].item()
        attr = deeplift.attribute(input_i, baselines=baseline_i, target=target_i)
        batch_relevances.append(attr.squeeze(0).detach().cpu().numpy())
    batch_relevances = np.stack(batch_relevances)
    all_relevances.append(batch_relevances)
    all_labels.append(labels_batch.detach().cpu().numpy())

all_relevances = np.concatenate(all_relevances, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

records = []
for cls in np.unique(all_labels):
    arr = np.mean(all_relevances[all_labels == cls], axis=0)
    top15 = arr.argsort()[-15:][::-1]
    bot15 = arr.argsort()[:15]
    for rank, g in enumerate(top15, 1):
        records.append((label_dict[cls], rank, gene_names[g], arr[g]))
    for rank, g in enumerate(bot15[::-1], 1):
        records.append((label_dict[cls], -rank, gene_names[g], arr[g]))

df = pd.DataFrame(records, columns=['class', 'rank', 'gene', 'value'])
os.makedirs(args.save_dir, exist_ok=True)
df.to_csv(os.path.join(args.save_dir, "deeplift_relevance_topbot.csv"), index=False)
