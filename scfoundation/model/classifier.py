import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from tqdm import tqdm

embedding_path = "/data1/data/corpus/scDATA/scfoundation/cell-anno_cell-anno_singlecell_cell_embedding_t4_resolution.npy"
LABEL_PATH = "/data1/data/corpus/scDATA/scfoundation/cell_type_labels.npy"
ANNOT_PATH = "/data1/data/corpus/scDATA/scfoundation/cell-anno.csv"

BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

X = np.load(embedding_path)
df = pd.read_csv(ANNOT_PATH)
df = df.loc[:, ["cell_type"]]
unique_classes = sorted(df["cell_type"].unique())
class_to_id = {cls: i for i, cls in enumerate(unique_classes)}
y = df["cell_type"].map(class_to_id).values
np.save(LABEL_PATH, y)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

class EmbeddingClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_dim, affine=False, eps=1e-6)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )
    def forward(self, x):
        x = self.norm(x)
        return self.fc(x)

model = EmbeddingClassifier(input_dim=X.shape[1], n_classes=len(unique_classes)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (pbar.n + 1))
    print(f"Epoch {epoch+1} loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        pbar_test = tqdm(test_loader, desc="Eval", leave=False)
        for xb, yb in pbar_test:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(yb.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    print(f"Epoch {epoch+1} Test Accuracy: {acc:.4f}")

print(classification_report(all_targets, all_preds, target_names=unique_classes))
