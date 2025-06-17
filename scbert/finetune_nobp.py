import os
import argparse
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from performer_pytorch import PerformerLM
import scanpy as sc
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import random
from utils import save_best_ckpt
from tqdm import tqdm


def get_reduced(tensor, current_device, dest_device, world_size):
    tensor = tensor.clone().detach() if torch.is_tensor(tensor) else torch.tensor(tensor)
    tensor = tensor.to(current_device)
    dist.reduce(tensor, dst=dest_device)
    if dist.get_rank() == dest_device:
        return tensor.item() / world_size
    else:
        return None

def distributed_concat(tensor, num_total_examples, world_size):
    local_size = torch.tensor([tensor.shape[0]], device=tensor.device)
    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)

    max_size = max([sz.item() for sz in size_list])
    padded_tensor = torch.cat([
        tensor,
        torch.zeros((max_size - tensor.shape[0], *tensor.shape[1:]), device=tensor.device)
    ], dim=0)

    output_tensors = [torch.zeros_like(padded_tensor) for _ in range(world_size)]
    dist.all_gather(output_tensors, padded_tensor)

    concat = torch.cat([out[:size.item()] for out, size in zip(output_tensors, size_list)], dim=0)
    return concat[:num_total_examples]

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, default=16906)
parser.add_argument("--epoch", type=int, default=2)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--grad_acc", type=int, default=60)
parser.add_argument("--valid_every", type=int, default=1)
parser.add_argument("--pos_embed", type=bool, default=True)
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad')
parser.add_argument("--model_path", type=str, default='./panglao_pretrained.pth')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/')
parser.add_argument("--model_name", type=str, default='finetune')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every
UNASSIGN_THRES = 0.0
PATIENCE = 10

model_name = args.model_name
ckpt_dir = args.ckpt_dir

local_rank = args.local_rank
is_master = local_rank == 0

dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = torch.distributed.get_world_size()

print(f"[Init] Seed: {SEED}, Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
print(f"[Init] Using {world_size} GPUs, local_rank: {local_rank}")

CLASS = args.bin_num + 2
SEQ_LEN = args.gene_num + 1

class SCDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __getitem__(self, index):
        full_seq = self.data[index].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0])))
        if index < 3 and is_master:
            print(f"[Dataset] idx {index}, sequence sample (first 10): {full_seq[:10]}, label: {self.label[index]}")
        return full_seq, self.label[index]
    def __len__(self):
        return self.data.shape[0]

class Identity(nn.Module):
    def __init__(self, dropout=0., h_dim=100, out_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(SEQ_LEN, out_dim)
        self.act1 = nn.ReLU()
        self._printed = False
    def forward(self, x):
        if not self._printed:
            print(f"Shape after Performer, before Identity: {x.shape}")
        x = x[:, None, :, :]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act1(x)
        if not self._printed:
            print(f"Shape after Identity: {x.shape}")
            self._printed = True
        return x

data = sc.read_h5ad(args.data_path)
label_dict, label = np.unique(data.obs['celltype'], return_inverse=True)
label = torch.from_numpy(label)
data = data.X

if is_master:
    os.makedirs(args.ckpt_dir, exist_ok=True)
    with open(os.path.join(args.ckpt_dir, 'label_dict.pkl'), 'wb') as f:
        pkl.dump(label_dict, f)
    with open(os.path.join(args.ckpt_dir, 'label.pkl'), 'wb') as f:
        pkl.dump(label, f)
    print(f"[Data] Loaded data shape: {data.shape}, Labels: {len(label_dict)}")

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
for train_idx, val_idx in sss.split(data, label):
    train_dataset = SCDataset(data[train_idx], label[train_idx])
    val_dataset = SCDataset(data[val_idx], label[val_idx])

if is_master:
    print(f"[Split] Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

model = PerformerLM(
    num_tokens=7,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    g2v_position_emb=args.pos_embed
)

ckpt = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
for param in model.parameters():
    param.requires_grad = False

model.to_out = Identity(dropout=0., h_dim=128, out_dim=len(label_dict))
model.add_module("to_out", model.to_out)
for param in model.to_out.parameters():
    param.requires_grad = True

model = model.to(device)
model = DDP(model, device_ids=[local_rank])

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if is_master:
    print(f"[Model] Loaded pretrained weights. Trainable params: {trainable_params}")

criterion = nn.CrossEntropyLoss().to(local_rank)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=0, pin_memory=True)
val_sampler = DistributedSampler(val_dataset)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=0, pin_memory=True) if is_master else None

dist.barrier()
trigger_times = 0
max_acc = 0.0


try:
    for i in range(1, EPOCHS + 1):
        train_loader.sampler.set_epoch(i)
        model.train()
        dist.barrier()

        running_loss = 0.0
        cum_acc = 0.0
        batch_count = 0

        if is_master:
            print(f"[Epoch {i}] Starting training...")

        with tqdm(train_loader, disable=not is_master, desc=f"Train Epoch {i}") as pbar:
            for data, labels in pbar:
                batch_count += 1
                data, labels = data.to(device), labels.to(device)

                try:
                    if batch_count % GRADIENT_ACCUMULATION != 0:
                        with model.no_sync():
                            logits = model(data)
                            loss = criterion(logits, labels)
                            loss.backward()
                    else:
                        logits = model(data)
                        loss = criterion(logits, labels)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
                        optimizer.step()
                        optimizer.zero_grad()

                    running_loss += loss.item()
                    softmax = nn.Softmax(dim=-1)
                    final = softmax(logits).argmax(dim=-1)
                    correct_num = torch.eq(final, labels).sum().item()
                    cum_acc += correct_num / labels.size(0)

                    if is_master:
                        pbar.set_postfix(loss=loss.item())
                except Exception as e:
                    print(f"[Error][Epoch {i}, Batch {batch_count}] {str(e)}")
                    continue

        epoch_loss = running_loss / batch_count
        epoch_acc = 100 * cum_acc / batch_count
        epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
        epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)

        if is_master:
            print(f'[Epoch {i}] Train Loss: {epoch_loss:.6f} | Train Accuracy: {epoch_acc:.2f}%')
        dist.barrier()

        do_val = (i % VALIDATE_EVERY == 0)
        if do_val:
            model.eval()

        dist.barrier()

        if do_val and is_master:
            print(f"[Epoch {i}] Starting validation...")
            running_loss = 0.0
            predictions, truths = [], []

            with tqdm(val_loader, disable=not is_master, desc=f"Val Epoch {i}") as val_pbar, torch.no_grad():
                for index, (data_v, labels_v) in enumerate(val_pbar):
                    try:
                        data_v, labels_v = data_v.to(device), labels_v.to(device)
                        logits = model(data_v)
                        loss = criterion(logits, labels_v)
                        running_loss += loss.item()

                        softmax = nn.Softmax(dim=-1)
                        final_prob = softmax(logits)
                        final = final_prob.argmax(dim=-1)
                        prob_np = final_prob.cpu().numpy()
                        mask = np.amax(prob_np, axis=-1) < UNASSIGN_THRES
                        final[torch.tensor(mask, device=final.device)] = -1
                        predictions.append(final)
                        truths.append(labels_v)
                    except Exception as e:
                        print(f"[Error][Val Epoch {i}, Batch {index}] {str(e)}")
                        continue

            try:
                predictions = torch.cat(predictions, dim=0)
                truths = torch.cat(truths, dim=0)
            except Exception as e:
                print(f"[Concatenation Error] {e}")
                continue

            try:
                predictions = distributed_concat(predictions, len(val_sampler.dataset), world_size)
                truths = distributed_concat(truths, len(val_sampler.dataset), world_size)
                no_drop = predictions != -1
                predictions = np.array(predictions[no_drop].cpu())
                truths = np.array(truths[no_drop].cpu())

                cur_acc = accuracy_score(truths, predictions)
                f1 = f1_score(truths, predictions, average='macro')
                val_loss = running_loss / len(val_loader)
                val_loss = get_reduced(val_loss, local_rank, 0, world_size)

                print(f'[Epoch {i}] Val Loss: {val_loss:.6f} | Val F1: {f1:.4f}')
                print(confusion_matrix(truths, predictions))
                print(classification_report(truths, predictions, target_names=label_dict.tolist(), digits=4))

                if cur_acc > max_acc:
                    max_acc = cur_acc
                    trigger_times = 0
                    save_best_ckpt(i, model, optimizer, val_loss, model_name, ckpt_dir)
                    print(f'[Epoch {i}] New best model saved (Acc: {cur_acc:.4f})')
                else:
                    trigger_times += 1
                    if trigger_times > PATIENCE:
                        print(f"[Early Stop] Triggered at Epoch {i}")
                        break
            except Exception as e:
                print(f"[Validation Error] {str(e)}")
            finally:
                del predictions, truths
        else:
            dist.barrier()
except KeyboardInterrupt:
    print("[Training Interrupted] Saving last checkpoint...")
    if is_master:
        save_best_ckpt(i, model, optimizer, epoch_loss, f"{model_name}_interrupted", ckpt_dir)