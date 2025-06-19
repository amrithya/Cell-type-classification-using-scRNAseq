import os
import csv
import copy
import torch
import shap
import traceback
import argparse
import numpy as np
import pandas as pd
import helper as h
import scanpy as sc
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class LRP:
    def __init__(self, model, rule='epsilon', epsilon=1e-7):
        self.model = model
        self.model.eval()
        self.epsilon = epsilon
        self.rule = rule

    def forward(self, x):
        self.local_input = x.clone().requires_grad_(True)
        return self.model(self.local_input)

    def relprop(self, x, R=None):
        if hasattr(self, 'local_input') and self.local_input.grad is not None:
            self.local_input.grad.zero_()
        self.local_input = x.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            output = self.model(self.local_input)
            if R is None:
                R = torch.zeros_like(output)
                R.scatter_(1, output.argmax(dim=1, keepdim=True), 1.0)
            output.backward(R, retain_graph=True)
            relevance = self.local_input.grad * self.local_input.data
            self.local_input.grad = None
        return relevance.detach()
    __call__ = relprop

class ClassLogitWrapper(torch.nn.Module):
    def __init__(self, model, target_class):
        super().__init__()
        self.model = model
        self.target_class = target_class

    def forward(self, x):
        out = self.model(x)
        return out[:, self.target_class].unsqueeze(1)

def train_nn(device, train_data, test_data, lr_rate, weights, input_size, output_size, dropout_rate, hidden_size):
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    class NNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, dropout_rate):
            super(NNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
            self.dropout2 = nn.Dropout(dropout_rate)
            self.fc3 = nn.Linear(hidden_size // 2, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            return x

    num_epochs = 10
    save_path = "/data1/data/corpus/scMODEL/shap_nn_model_Zheng68K.pth"

    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        model = NNet(input_size, hidden_size, output_size, dropout_rate).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded existing model from {save_path}")
        model.eval()

        train_correct, train_total = 0, 0
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct_mask = (predicted == labels)
                train_correct += correct_mask.sum().item()
                train_total += labels.size(0)
                
        train_accuracy = train_correct / train_total * 100

        test_correct, test_total = 0, 0
        test_correct_indices = []
        sample_index = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct_mask = (predicted == labels)
                test_correct += correct_mask.sum().item()
                test_total += labels.size(0)
                batch_indices = torch.arange(sample_index, sample_index + labels.size(0))[correct_mask.cpu()]
                test_correct_indices.extend(batch_indices.tolist())
                sample_index += labels.size(0)
        test_accuracy = test_correct / test_total * 100

        print(f"Input size {input_size}, Hidden size {hidden_size}, learning rate {lr_rate}, dropout {dropout_rate}=> Train Accuracy:{train_accuracy:.2f} :: Test Accuracy: {test_accuracy:.2f}%")
        return model, test_accuracy, train_accuracy, test_correct_indices

    else:
        model = NNet(input_size, hidden_size, output_size, dropout_rate).to(device)
        weights = weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(model.parameters(), lr=lr_rate)

        print(f"\nTraining with Hidden size: {hidden_size},learning rate: {lr_rate}, dropout rate: {dropout_rate}")
        l1_lambda = 1e-5
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct / total * 100
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        model.eval()
        correct = 0
        total = 0
        correct_indices = []
        sample_index = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct_mask = (predicted == labels)
                correct += correct_mask.sum().item()
                total += labels.size(0)
                batch_indices = torch.arange(sample_index, sample_index + labels.size(0))[correct_mask.cpu()]
                correct_indices.extend(batch_indices.tolist())
                sample_index += labels.size(0)
        test_accuracy = correct / total * 100

        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'dropout_rate': dropout_rate,
            'lr_rate': lr_rate,
            'weights': weights.cpu(),
        }, save_path)
        print(f"Model saved to {save_path}")
        print(f"Input size {input_size}, Hidden size {hidden_size}, learning rate {lr_rate}, dropout {dropout_rate}=> Train Accuracy:{epoch_accuracy:.2f} :: Test Accuracy: {test_accuracy:.2f}%")
        return model, test_accuracy, epoch_accuracy, correct_indices
    
def train_gru(device, train_data, test_data, lr_rate, weights, input_size, output_size, dropout_rate, hidden_size):
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    class GRUNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, dropout_rate):
            super(GRUNet, self).__init__()
            self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=dropout_rate)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, h_n = self.gru(x)
            out = h_n[-1]
            out = self.fc(out)
            return out

    def evaluate(loader):
        correct, total = 0, 0
        correct_indices = []
        sample_index = 0
        total_loss = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if inputs.dim() == 2:
                    inputs = inputs.unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_mask = (predicted == labels)
                correct += correct_mask.sum().item()
                total += labels.size(0)
                batch_indices = torch.arange(sample_index, sample_index + labels.size(0))[correct_mask.cpu()]
                correct_indices.extend(batch_indices.tolist())
                sample_index += labels.size(0)
        accuracy = correct / total * 100
        avg_loss = total_loss / len(loader)
        return accuracy, correct_indices, avg_loss

    model = GRUNet(input_size, hidden_size, output_size, dropout_rate).to(device)
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr_rate, weight_decay=1e-3)

    num_epochs = 25
    patience = 10
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    l1_lambda = 1e-5

    print(f"\nTraining with Hidden size: {hidden_size}, learning rate: {lr_rate}, dropout rate: {dropout_rate}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total * 100

        model.eval()
        val_acc, correct_indices, val_loss = evaluate(test_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_acc, final_correct_indices, _ = evaluate(test_loader)
    print(f"Final Test Accuracy after early stopping: {final_acc:.2f}%")

    return model, final_acc, epoch_accuracy, final_correct_indices


def analyze_lrp_classwise(model, lrp, X_test, y_test, test_correct_indices, gene_names, le, device):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f"nn_top_bottom15_genes_all_classes_from_lrp.csv")

    num_classes = int(y_test.max() + 1)
    class_names = le.inverse_transform(np.arange(num_classes))
    feature_dim = gene_names.shape[0]
    class_relevance = torch.zeros((num_classes, feature_dim), device=device)
    class_counts = torch.zeros(num_classes, dtype=torch.int)

    for idx in test_correct_indices:
        sample_slice = X_test[idx:idx+1]
        if issparse(sample_slice):
            sample_slice = sample_slice.toarray()

        label = y_test[idx]
        sample_input = torch.tensor(sample_slice, dtype=torch.float32, requires_grad=True).to(device)
        if len(sample_input.shape) == 1:
            sample_input = sample_input.unsqueeze(0)

        with torch.enable_grad():
            output = model(sample_input)
            relevance_scores = lrp(sample_input).squeeze(0)

        class_relevance[label] += relevance_scores
        class_counts[label] += 1

    for c in range(num_classes):
        if class_counts[c] > 0:
            class_relevance[c] /= class_counts[c]

    records = []

    for c in range(num_classes):
        relevance = class_relevance[c].cpu().numpy()
        top_indices = np.argsort(relevance)[-15:][::-1]
        bottom_indices = np.argsort(relevance)[:15]

        top_genes = [gene_names[i] for i in top_indices]
        bottom_genes = [gene_names[i] for i in bottom_indices]
        class_name = class_names[c]

        for i in range(15):
            records.append({
                "class": class_name,
                "rank": i + 1,
                "top_gene": top_genes[i],
                "top_score": relevance[top_indices[i]],
                "bottom_gene": bottom_genes[i],
                "bottom_score": relevance[bottom_indices[i]]
            })

    df = pd.DataFrame(records)
    if os.path.exists(results_file):
        os.remove(results_file)
    df.to_csv(results_file, mode='a', header=not os.path.exists(results_file), index=False)


def shap_explain_nn(model, test_data, feature_names, le, device, save_name='nn'):
    print(f"Explaining PyTorch model predictions using SHAP (GradientExplainer) for model {save_name}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{save_name}_top_bottom15_genes_all_classes_from_shap.csv")

    X_test = test_data.tensors[0].cpu().numpy()
    y_test = test_data.tensors[1].cpu().numpy()
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    model.eval()

    with torch.no_grad():
        X_tensor_all = torch.tensor(X_test, dtype=torch.float32).to(device)
        print("X_tensor_all shape:", X_tensor_all.shape)

        outputs = model(X_tensor_all)
        print("outputs shape:", outputs.shape)

        _, predicted = torch.max(outputs, 1)
        print("predicted shape:", predicted.shape)

    correct_mask = (predicted.cpu().numpy() == y_test)
    print("correct_mask shape:", correct_mask.shape)

    X_correct = X_test[correct_mask]
    y_correct = y_test[correct_mask]
    print("X_correct shape:", X_correct.shape)
    print("y_correct shape:", y_correct.shape)

    if X_correct.shape[0] > 10:
        X_correct = X_correct[:10]
        y_correct = y_correct[:10]
        print("X_correct shape after slicing:", X_correct.shape)
        print("y_correct shape after slicing:", y_correct.shape)

    X_tensor = torch.tensor(X_correct, dtype=torch.float32).to(device)
    print("X_tensor shape (for SHAP):", X_tensor.shape)

    num_classes = model(X_tensor).shape[1]
    print("num_classes:", num_classes)

    feature_dim = len(feature_names)
    print("feature_dim (number of features):", feature_dim)

    records = []

    for target_class in range(num_classes):
        print(f"Explaining class {target_class}...")

        wrapped_model = ClassLogitWrapper(model, target_class).to(device)

        explainer = shap.GradientExplainer(wrapped_model, X_tensor)
        shap_values = explainer.shap_values(X_tensor)

        print(f"shap_values type: {type(shap_values)}")
        print(f"shap_values[0] shape: {np.array(shap_values[0]).shape}")

        shap_vals = shap_values[0]

        mean_shap = np.mean(shap_vals, axis=0)
        print("mean_shap shape:", mean_shap.shape)

        top_indices = np.argsort(mean_shap)[-15:][::-1]
        bottom_indices = np.argsort(mean_shap)[:15]
        print("top_indices shape:", top_indices.shape)
        print("bottom_indices shape:", bottom_indices.shape)

        top_genes = [feature_names[i] for i in top_indices]
        bottom_genes = [feature_names[i] for i in bottom_indices]
        print("Number of top_genes:", len(top_genes))
        print("Number of bottom_genes:", len(bottom_genes))

        if target_class < len(le.classes_):
            class_name = le.inverse_transform([target_class])[0]
        else:
            class_name = f"class_{target_class}"

        num_top = min(15, len(top_genes))
        for i in range(num_top):
            records.append({
                "class": class_name,
                "rank": i + 1,
                "top_gene": top_genes[i],
                "top_score": mean_shap[top_indices[i]],
                "bottom_gene": bottom_genes[i],
                "bottom_score": mean_shap[bottom_indices[i]]
            })

    df = pd.DataFrame(records)
    print("df shape (final result):", df.shape)

    if os.path.exists(results_file):
        os.remove(results_file)
    df.to_csv(results_file, index=False)
    print(f"Saved SHAP results to {results_file}")
