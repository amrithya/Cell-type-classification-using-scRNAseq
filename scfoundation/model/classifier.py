import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import scanpy as sc
from tqdm import tqdm
import shap

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
    return labels_encoded, le, adata

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

def interpret_with_shap(model, embeddings, adata, background_samples=100, n_genes=15):
    gene_names = np.array(adata.var_names)

    device = next(model.parameters()).device
    model.eval()

    background = embeddings[np.random.choice(len(embeddings), background_samples, replace=False)]
    explainer = shap.DeepExplainer(model, torch.tensor(background).float().to(device))
    
    test_samples = torch.tensor(embeddings[:1000]).float().to(device)
    shap_values = explainer.shap_values(test_samples)

    if isinstance(shap_values, list):
        mean_shap = np.mean(np.array(shap_values), axis=(0, 1))
    else:
        mean_shap = np.mean(shap_values, axis=0)

    sorted_dims = np.argsort(mean_shap)
    top_dims = sorted_dims[-n_genes:]
    bottom_dims = sorted_dims[:n_genes]

    token_weights = model.fc[0].weight.detach().cpu().numpy()

    def get_gene_scores(dimensions):
        scores = np.zeros(len(gene_names))
        for dim in dimensions:
            scores += token_weights[:, dim].mean() * mean_shap[dim]
        return scores

    top_scores = get_gene_scores(top_dims)
    bottom_scores = get_gene_scores(bottom_dims)

    top_genes = gene_names[np.argsort(top_scores)[-n_genes:]]
    bottom_genes = gene_names[np.argsort(bottom_scores)[:n_genes]]

    return {
        'top_genes': top_genes,
        'bottom_genes': bottom_genes,
        'shap_values': mean_shap,
        'top_dims': top_dims,
        'bottom_dims': bottom_dims
    }

def main():
    embedding_path = '/data1/data/corpus/scDATA/scfoundation/cell-anno_cell-anno_singlecell_cell_embedding_t4_resolution.npy'
    label_path = '/data1/data/corpus/scDATA/scfoundation/Zheng68K_foundation.h5ad'

    embeddings = np.load(embedding_path)
    labels, label_encoder, adata = load_labels(label_path)
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

    shap_results = interpret_with_shap(model, embeddings, adata, background_samples=100, n_genes=15)
    print("\nTop 15 Genes (SHAP):")
    print(shap_results['top_genes'])
    print("\nBottom 15 Genes (SHAP):")
    print(shap_results['bottom_genes'])

if __name__ == '__main__':
    main()
