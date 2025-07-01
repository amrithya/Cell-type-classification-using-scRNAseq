import sys
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

sys.path.append("../model/")
from load import *

class GeneExpressionDataset(Dataset):
    def __init__(self, csv_path, label_col='label'):
        df = pd.read_csv(csv_path, index_col=0)
        labels_int, uniques = pd.factorize(df[label_col].values)
        self.labels = torch.tensor(labels_int, dtype=torch.long)
        self.features = torch.tensor(df.drop(columns=[label_col]).values, dtype=torch.float32)
        self.label_map = {i: label for i, label in enumerate(uniques)}

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

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch['x'].to(device)
            labels = batch['targets'].to(device)
            outputs = model({'x': inputs, 'targets': labels})
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_df = pd.read_csv('/data1/data/corpus/scDATA/Zheng68K_scf_preprocessed.csv', index_col=0)
    train_df, test_df = train_test_split(full_df, test_size=0.2, stratify=full_df['label'], random_state=42)
    train_df.to_csv('train_temp.csv')
    test_df.to_csv('test_temp.csv')

    train_dataset = GeneExpressionDataset(csv_path='train_temp.csv', label_col='label')
    test_dataset = GeneExpressionDataset(csv_path='test_temp.csv', label_col='label')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_classes = len(train_dataset.label_map)
    model = LinearProbingClassifier(ckpt_path='/data1/data/corpus/scMODEL/scfoundation/models.ckpt', n_class=num_classes)
    model.build()
    model = model.to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        total_loss = 0
        print(f"\nEpoch {epoch+1}")
        pbar = tqdm(train_loader, desc=f"Training", leave=False)
        for batch in pbar:
            inputs = batch['x'].to(device)
            labels = batch['targets'].to(device)
            optimizer.zero_grad()
            outputs = model({'x': inputs, 'targets': labels})
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(train_loader):.4f}")

    save_path = '/data1/data/corpus/scMODEL/scfoundation/scfoundation_model.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()
