import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import scanpy as sc
import shap
from tqdm import tqdm

class CellEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        print(f"[DEBUG] Creating dataset with {len(labels)} samples")
        self.embeddings = embeddings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx]).float(), self.labels[idx]

class LinearProbingClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        print(f"[DEBUG] Initializing model with input dim: {input_dim}, num classes: {num_classes}")
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
    print(f"[DEBUG] Loading labels from {path}")
    adata = sc.read_h5ad(path)
    labels = adata.obs['celltype'].values
    print(f"[DEBUG] Loaded {len(labels)} labels")
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    print(f"[DEBUG] Encoded labels into {len(np.unique(labels_encoded))} classes")
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
    print(f"[DEBUG] Training - Loss: {total_loss:.4f}, Accuracy: {total_correct/total:.4f}")
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
    print(f"[DEBUG] Evaluation - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    return avg_loss, avg_acc, all_preds, all_targets

def interpret_with_shap(shap_layer, embeddings, adata, token_weights, background_samples=100, n_genes=15):
    print(f"[DEBUG] Running SHAP interpretation with {background_samples} background samples")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    background = torch.tensor(embeddings[np.random.choice(len(embeddings), background_samples, replace=False)]).float().to(device)
    explainer = shap.DeepExplainer(shap_layer, background)
    full_input = torch.tensor(embeddings).float().to(device)
    shap_values = explainer.shap_values(full_input)
    print(f"[DEBUG] SHAP values shape: {np.array(shap_values[0]).shape}")
    mean_shap = np.mean(shap_values[0], axis=0)
    scores = np.zeros(mean_shap.shape[0])
    for dim in tqdm(range(token_weights.shape[1]), desc="Aggregating SHAP scores"):
        scores += token_weights[:, dim].mean() * mean_shap
    top_gene_ids = np.argsort(scores)[-n_genes:][::-1]
    bottom_gene_ids = np.argsort(scores)[:n_genes]
    top_gene_names = adata.var_names[top_gene_ids].tolist()
    bottom_gene_names = adata.var_names[bottom_gene_ids].tolist()
    return {
        'top_genes': top_gene_names,
        'bottom_genes': bottom_gene_names,
        'top_scores': scores[top_gene_ids].tolist(),
        'bottom_scores': scores[bottom_gene_ids].tolist()
    }

def main():
    embedding_path = '/data1/data/corpus/scDATA/scfoundation/mt_kidney_rcc_cca_paa_2_scf.npy'
    label_path = '/data1/data/corpus/scDATA/cancer/data/mt_kidney_rcc_cca_paa_2/grade/mt_kidney_rcc_cca_paa_2.h5ad'

    print("[DEBUG] Loading embeddings and labels")
    embeddings = np.load(embedding_path)
    labels, label_encoder, adata = load_labels(label_path)
    print(f"[DEBUG] Embedding shape: {embeddings.shape}")
    assert embeddings.shape[0] == labels.shape[0], "[ERROR] Mismatch in embedding and label count"

    train_x, test_x, train_y, test_y = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels)
    print(f"[DEBUG] Train size: {len(train_y)}, Test size: {len(test_y)}")

    train_dataset = CellEmbeddingDataset(train_x, train_y)
    test_dataset = CellEmbeddingDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEBUG] Using device: {device}")
    model = LinearProbingClassifier(input_dim=embeddings.shape[1], num_classes=len(label_encoder.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 20
    for epoch in range(epochs):
        print(f"[DEBUG] Starting epoch {epoch + 1}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f} Acc: {train_acc:.4f}")

    val_loss, val_acc, all_preds, all_targets = eval_epoch(model, test_loader, criterion, device)
    print(f"Final Eval - Val loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    report = classification_report(all_targets, all_preds, target_names=label_encoder.classes_)
    print("Classification Report:\n", report)

    print("[DEBUG] Extracting first FC layer weights for SHAP interpretation")
    token_weights = model.fc[0].weight.detach().cpu().numpy()
    # shap_results = interpret_with_shap(model.fc[0], embeddings, adata, token_weights, background_samples=100, n_genes=15)

if __name__ == '__main__':
    main()
