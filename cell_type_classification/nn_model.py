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
    print(f"Explaining PyTorch model predictions using SHAP for model {save_name}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{save_name}_top_bottom15_genes_all_classes_from_shap.csv")
    
    X_test = test_data.tensors[0].cpu().numpy()
    y_test = test_data.tensors[1].cpu().numpy()

    X_tensor = torch.tensor(X_correct, dtype=torch.float32).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
    correct_mask = (predicted.cpu().numpy() == y_test)
    X_correct = X_test[correct_mask]
    y_correct = y_test[correct_mask]
    print(f"Number of correctly predicted test samples: {X_correct.shape[0]}")

    explainer = shap.GradientExplainer(model, X_tensor)

    shap_values = explainer.shap_values(X_tensor)

    num_classes = len(shap_values)
    class_names = le.inverse_transform(np.arange(num_classes))
    feature_dim = len(feature_names)

    class_relevance = np.zeros((num_classes, feature_dim))
    class_counts = np.zeros(num_classes, dtype=int)

    for i in range(X_correct.shape[0]):
        label = y_correct[i]
        class_relevance[label] += shap_values[label][i]
        class_counts[label] += 1

    for c in range(num_classes):
        if class_counts[c] > 0:
            class_relevance[c] /= class_counts[c]

    records = []
    for c in range(num_classes):
        relevance = class_relevance[c]
        top_indices = np.argsort(relevance)[-15:][::-1]
        bottom_indices = np.argsort(relevance)[:15]
        
        top_genes = [feature_names[i] for i in top_indices]
        bottom_genes = [feature_names[i] for i in bottom_indices]
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
    df.to_csv(results_file, index=False)
    print(f"Saved SHAP results to {results_file}")
