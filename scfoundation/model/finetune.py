import sys
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
sys.path.append("../model/")
from load import *

class GeneExpressionDataset(Dataset):
    def __init__(self, csv_path, label_col='label'):
        df = pd.read_csv(csv_path, index_col=0)
        
        self.labels = torch.tensor(df[label_col].values, dtype=torch.long)
        self.features = torch.tensor(df.drop(columns=[label_col]).values, dtype=torch.float32)
        
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        return {
            'x': self.features[idx],
            'targets': self.labels[idx]
        }

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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = GeneExpressionDataset(csv_path='/data1/data/corpus/scDATA/Zheng68K_scf_preprocessed.csv', label_col='label')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LinearProbingClassifier(ckpt_path='/data1/data/corpus/scMODEL/scfoundation/models.ckpt', n_class=11)
    model.build()
    model = model.to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in dataloader:
            inputs = batch['x'].to(device)
            labels = batch['targets'].to(device)

            optimizer.zero_grad()
            outputs = model({'x': inputs, 'targets': labels})

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

if __name__ == '__main__':
    main()
