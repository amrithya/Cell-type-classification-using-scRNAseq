import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import scanpy as sc

sys.path.append("../model/")
from load import *

class Zheng68KDataset(Dataset):
    def __init__(self, h5ad_path):
        self.adata = sc.read_h5ad(h5ad_path)
        self.X = self.adata.X.toarray() if not isinstance(self.adata.X, np.ndarray) else self.adata.X
        if 'celltype' in self.adata.obs:
            self.y = self.adata.obs['celltype'].astype('category').cat.codes.values
        else:
            raise ValueError("No 'celltype' column found in adata.obs!")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return {'x': x, 'targets': y}

def bin_gene_expression(x, n_bins, max_expr=10.0):
    x = torch.clamp(x, 0, max_expr)
    bin_width = max_expr / n_bins
    x_binned = (x / bin_width).long()
    x_binned = torch.clamp(x_binned, 0, n_bins - 1)
    return x_binned

class LinearProbingClassifier(nn.Module):
    def __init__(self, ckpt_path, n_classes, n_bins=100, frozenmore=True):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.n_classes = n_classes
        self.n_bins = n_bins
        self.frozenmore = frozenmore

    def build(self):
        model, model_config = load_model_frommmf(self.ckpt_path)
        self.model_config = model_config
        hidden_dim = model_config['encoder']['hidden_dim']
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder
        if self.frozenmore:
            self._freeze_module(self.token_emb)
            self._freeze_module(self.pos_emb)
        self._freeze_module(self.encoder)
        if hasattr(self.encoder, 'transformer_encoder'):
            for name, param in self.encoder.transformer_encoder[-2].named_parameters():
                param.requires_grad = True
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_classes)
        )
        self.norm = nn.BatchNorm1d(hidden_dim, affine=False, eps=1e-6)

    def forward(self, sample_list):
        x_cont = sample_list['x']
        x_binned = bin_gene_expression(x_cont, self.n_bins)
        value_mask = x_binned > 0
        x_binned, x_padding = gatherData(x_binned, value_mask, self.model_config['pad_token_id'])
        gene_ids = torch.arange(x_binned.shape[1], device=x_binned.device).unsqueeze(0).expand(x_binned.shape[0], -1)
        value_mask_new = torch.ones_like(x_binned, dtype=torch.bool)
        pos_ids, _ = gatherData(gene_ids, value_mask_new, self.model_config['pad_token_id'])
        x_emb = self.token_emb(x_binned, output_weight=0)
        pos_emb = self.pos_emb(pos_ids)
        x_emb = x_emb + pos_emb
        logits = self.encoder(x_emb, x_padding)
        logits, _ = torch.max(logits, dim=1)
        logits = self.norm(logits)
        logits = self.fc1(logits)
        return logits

    @staticmethod
    def _freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False

if __name__ == '__main__':
    ckpt_path = '/data1/data/corpus/scMODEL/scfoundation/models.ckpt'
    data_path = '/data1/data/corpus/scDATA/Zheng68K.h5ad'
    dataset = Zheng68KDataset(data_path)
    n_classes = len(np.unique(dataset.y))
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    model = LinearProbingClassifier(ckpt_path, n_classes, n_bins=100)
    model.build()
    model = model.cuda()
    for batch in loader:
        batch['x'] = batch['x'].cuda()
        batch['targets'] = batch['targets'].cuda()
        logits = model(batch)
        print("Logits shape:", logits.shape)
        break
