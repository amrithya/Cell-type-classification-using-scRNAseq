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
        print(f"Loading dataset from {h5ad_path}")
        self.adata = sc.read_h5ad(h5ad_path)
        self.X = self.adata.X.toarray() if not isinstance(self.adata.X, np.ndarray) else self.adata.X
        self.y = self.adata.obs['celltype'].astype('category').cat.codes.values
        print(f"Loaded {self.X.shape[0]} cells, {self.X.shape[1]} genes, {len(np.unique(self.y))} classes")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return {'x': x, 'targets': y}

class LinearProbingClassifier(nn.Module):
    def __init__(self, ckpt_path, n_classes, frozenmore=True):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.frozenmore = frozenmore
        self.n_classes = n_classes

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
            print('self.pos_emb and self.token_emb frozen')

        self._freeze_module(self.encoder)

        if hasattr(self.encoder, 'transformer_encoder'):
            target_layer = self.encoder.transformer_encoder[-2]
            for name, param in target_layer.named_parameters():
                param.requires_grad = True
                print(f'self.encoder.transformer_encoder[-2] {name} has grad')

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_classes)
        )
        self.norm = nn.BatchNorm1d(hidden_dim, affine=False, eps=1e-6)

    def forward(self, sample_list):
        x = sample_list['x']  # (B, L)
        value_mask = x > 0
        x, x_padding = gatherData(x, value_mask, self.model_config['pad_token_id'])

        num_genes = x.shape[1]
        gene_ids = torch.arange(num_genes, device=x.device).repeat(x.shape[0], 1)
        position_ids, _ = gatherData(gene_ids, value_mask, self.model_config['pad_token_id'])

        x_emb = self.token_emb(torch.unsqueeze(x, 2), output_weight=0)
        pos_emb = self.pos_emb(position_ids)
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

    model = LinearProbingClassifier(ckpt_path, n_classes)
    model.build()
    model = model.cuda()

    for batch in loader:
        batch['x'] = batch['x'].cuda()
        batch['targets'] = batch['targets'].cuda()
        logits = model(batch)
        print("Logits shape:", logits.shape)
        break
