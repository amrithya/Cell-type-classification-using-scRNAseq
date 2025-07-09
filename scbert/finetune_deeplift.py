import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader
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

print("Loading data...")
data = sc.read_h5ad(args.data_path)
index_labels = data.obs['celltype']
label_dict, label = np.unique(np.array(index_labels), return_inverse=True)
data_counts = data.X
gene_names = data.var_names

print("Performing stratified split...")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
for _, index_val in sss.split(data_counts, label):
    data_val, label_val = data_counts[index_val], label[index_val]

print(f"Validation set size: {len(label_val)}")

class SCDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        full_seq = self.data[idx].toarray()[0]
        # Clip values to max token id
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq_long = torch.from_numpy(full_seq).long()
        full_seq_float = torch.from_numpy(full_seq).float()
        full_seq_long = torch.cat((full_seq_long, torch.tensor([0], dtype=torch.long)))
        full_seq_float = torch.cat((full_seq_float, torch.tensor([0.0], dtype=torch.float)))
        label = self.labels[idx]
        return full_seq_long, full_seq_float, label

val_dataset = SCDataset(data_val, label_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Validation DataLoader prepared.")

print("Loading model...")
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
print("Model loaded and set to eval mode.")

class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_float):
        # Convert float input to long tokens internally
        input_long = input_float.round().long()
        logits = self.model(x=input_long)
        return logits[:, -1, :]

wrapped_model = WrappedModel(model).to(device)
wrapped_model.eval()

deeplift = DeepLift(wrapped_model)
print("DeepLift initialized.")

all_relevances = []
all_labels = []

print("Starting DeepLift attribution...")
for batch_idx, (inputs_long, inputs_float, labels_batch) in enumerate(tqdm(val_loader)):
    print(f"\nBatch {batch_idx + 1} --------------------")
    print(f"inputs_long shape: {inputs_long.shape}, dtype: {inputs_long.dtype}")
    print(f"inputs_float shape: {inputs_float.shape}, dtype: {inputs_float.dtype}")
    print(f"labels_batch shape: {labels_batch.shape}, dtype: {labels_batch.dtype}")

    inputs_float = inputs_float.to(device)
    labels_batch = labels_batch.to(device)
    baseline = torch.zeros_like(inputs_float).to(device)

    batch_relevances = []
    for i in range(inputs_float.size(0)):
        input_float_i = inputs_float[i].unsqueeze(0)
        baseline_i = baseline[i].unsqueeze(0)
        target_i = int(labels_batch[i].item())

        print(f"\n Sample {i}:")
        print(f"input_float_i shape: {input_float_i.shape}, dtype: {input_float_i.dtype}")
        print(f"baseline_i shape: {baseline_i.shape}, dtype: {baseline_i.dtype}")
        print(f"target_i: {target_i}, type: {type(target_i)}")

        with torch.no_grad():
            out = wrapped_model(input_float_i)
            print(f"logits shape: {out.shape}, dtype: {out.dtype}")
            if target_i >= out.shape[1]:
                print(f"Skipping sample due to invalid target: {target_i}, output shape: {out.shape}")
                continue

        input_float_i.requires_grad_()
        attr = deeplift.attribute(input_float_i, baselines=baseline_i, target=target_i)
        print(f"attr shape: {attr.shape}, dtype: {attr.dtype}")
        batch_relevances.append(attr.squeeze(0).detach().cpu().numpy())

    if batch_relevances:
        batch_relevances = np.stack(batch_relevances)
        print(f"\nbatch_relevances shape after stack: {batch_relevances.shape}, dtype: {batch_relevances.dtype}")
        all_relevances.append(batch_relevances)
        all_labels.append(labels_batch.detach().cpu().numpy())
        print(f"Added to all_relevances and all_labels.")
    print(f"Processed batch {batch_idx + 1}")

print("All batches processed. Aggregating results...")
all_relevances = np.concatenate(all_relevances, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

records = []
print("Generating top/bottom genes per class...")
for cls in np.unique(all_labels):
    arr = np.mean(all_relevances[all_labels == cls], axis=0)
    top15 = arr.argsort()[-16:-1][::-1]
    bot15 = arr.argsort()[:15]
    for rank, g in enumerate(top15, 1):
        records.append((label_dict[cls], rank, gene_names[g], arr[g]))
    for rank, g in enumerate(bot15[::-1], 1):
        records.append((label_dict[cls], -rank, gene_names[g], arr[g]))
print("Done collecting feature relevance.")

os.makedirs(args.save_dir, exist_ok=True)
df = pd.DataFrame(records, columns=['class', 'rank', 'gene', 'value'])
save_path = os.path.join(args.save_dir, "deeplift_relevance_topbot.csv")
df.to_csv(save_path, index=False)
print(f"Saved relevance scores to: {save_path}")
