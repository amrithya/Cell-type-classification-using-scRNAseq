import sys
import numpy as np
import torch
from torch import nn
sys.path.append("../model/")
from load import *

class LinearProbingClassifier(nn.Module):

    def __init__(self, ckpt_path, frozenmore=True, n_class=10):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.frozenmore = frozenmore
        self.n_class = n_class

    def build(self):
        model, model_config = load_model_frommmf(self.ckpt_path)
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder

        if self.frozenmore:
            for _, p in self.token_emb.named_parameters():
                p.requires_grad = False
            for _, p in self.pos_emb.named_parameters():
                p.requires_grad = False
        
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False

        for na, param in self.encoder.transformer_encoder[-2:].named_parameters():
            print('self.encoder.transformer_encoder ', na, ' have grad')
            param.requires_grad = True

        hidden_dim = model_config['encoder']['hidden_dim']

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_class)
        )
        self.norm = torch.nn.BatchNorm1d(hidden_dim, affine=False, eps=1e-6)

        self.model_config = model_config

    def forward(self, sample_list, *args, **kwargs):
        label = sample_list['targets']

        x = sample_list['x']
        value_labels = x > 0
        x, x_padding = gatherData(x, value_labels, self.model_config['pad_token_id'])

        data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels, self.model_config['pad_token_id'])

        x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
        position_emb = self.pos_emb(position_gene_ids)
        x += position_emb

        logits = self.encoder(x, x_padding)

        logits, _ = torch.max(logits, dim=1)

        logits = self.norm(logits)
        logits = self.fc1(logits)

        return logits


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    finetune_model = LinearProbingClassifier(ckpt_path='./models/models.ckpt', n_class=10)
    finetune_model.build()
    finetune_model = finetune_model.to(device)

    sample_list = {
        'x': torch.zeros([8, 18264], device=device),
        'targets': torch.randint(0, 10, (8,), device=device)
    }
    sample_list['x'][:, :100] = 1

    output_logits = finetune_model(sample_list)
    print(output_logits.shape)
