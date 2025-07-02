import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import scanpy as sc
from tqdm import tqdm

class CellEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx]).float(), self.labels[idx]

class LinearProbingClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_dim, affine=False, eps=1e-6)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.norm(x)
        logits = self.fc(x)
        return logits
    
def load_labels(path):
    adata = sc.read_h5ad(path)
    labels = adata.obs['celltype'].values
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    return labels_encoded, le

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total = 0
    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total += x.size(0)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
    avg_loss = total_loss / total
    avg_acc = total_correct / total
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    return avg_loss, avg_acc, all_preds, all_targets

def main():
    embedding_path = '/data1/data/corpus/scDATA/scfoundation/cell-anno_cell-anno_singlecell_cell_embedding_t4_resolution.npy'
    label_path = '/data1/data/corpus/scDATA/scfoundation/Zheng68K_foundation.h5ad'

    embeddings = np.load(embedding_path)
    labels, label_encoder = load_labels(label_path)
    assert embeddings.shape[0] == labels.shape[0]

    train_x, test_x, train_y, test_y = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels)

    train_dataset = CellEmbeddingDataset(train_x, train_y)
    test_dataset = CellEmbeddingDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LinearProbingClassifier(input_dim=embeddings.shape[1], num_classes=len(label_encoder.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 20
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f} Acc: {train_acc:.4f}")

    val_loss, val_acc, all_preds, all_targets = eval_epoch(model, test_loader, criterion, device)
    print(f"Final Eval - Val loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    report = classification_report(all_targets, all_preds, target_names=label_encoder.classes_)
    print("Classification Report:\n", report)

if __name__ == '__main__':
    main()
