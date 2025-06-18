import os
import torch
import shap
from tqdm import tqdm
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import seaborn as sns
import xgboost as xgb
import torch.nn as nn
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from xgboost import XGBClassifier
from scipy.sparse import issparse
import scipy.sparse
import csv
import argparse
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score 
import helper as h



def shap_explain(clf, X_test, y_test, feature_names):
    """
    Function to explain model predictions using SHAP and save results.

    Returns:
    - shap_values: SHAP values for the test data
    - model: Trained model
    - explainer: SHAP explainer object
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("Explaining model predictions using SHAP for one correctly predicted sample")

    explainer = shap.Explainer(clf, X_test)
    y_pred = clf.predict(X_test)

    correct_indices = np.where(y_pred == y_test)[0]
    if len(correct_indices) == 0:
        raise ValueError("No correctly predicted samples found in test set.")
    sample_idx = correct_indices[0]

    shap_values = explainer(X_test[sample_idx].reshape(1, -1))

    if hasattr(shap_values, "values"):
        vals = shap_values.values
        print(f"shap_values.values shape: {vals.shape}")
        pred_class = y_pred[sample_idx]
        if vals.ndim == 3:
            num_samples = vals.shape[0]
            if vals.shape[1] == len(feature_names):
                sample_shap_values = vals[sample_idx, :, pred_class]
            elif vals.shape[2] == len(feature_names):
                sample_shap_values = vals[sample_idx, pred_class, :]
            else:
                raise ValueError("Unexpected shape of SHAP values for multiclass")
        elif vals.ndim == 2:
            sample_shap_values = vals[sample_idx]
        else:
            raise ValueError(f"Unexpected SHAP values ndim: {vals.ndim}")
    else:
        raise ValueError("SHAP values object has no attribute 'values'")

    df_shap = pd.DataFrame({
        'feature': feature_names,
        'shap_value': sample_shap_values
    }).sort_values(by='shap_value', key=abs, ascending=False)

    csv_path = os.path.join(results_dir, f'shap_values_sample_{sample_idx}_class_{y_pred[sample_idx]}.csv')
    df_shap.to_csv(csv_path, index=False)
    print(f"Saved SHAP values for sample {sample_idx} to {csv_path}")

    return shap_values, explainer


def shap_explain_all(clf, X_test, y_test, feature_names, le):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("Explaining model predictions using SHAP for all correctly samples")

    explainer = shap.Explainer(clf, X_test)
    y_pred = clf.predict(X_test)

    correct_indices = np.where(y_pred == y_test)[0]    
    if len(correct_indices) == 0:
        raise ValueError("No correctly predicted samples found in test set.")
    
    X_correct = X_test[correct_indices]
    correct_labels = y_test[correct_indices]

    print("Shape of correct_labels:",correct_labels.shape)

    count_class = {i: (correct_labels == i).sum() for i in range(11)}
    print("Counts of labels 0 to 10:")
    for label, count in count_class.items():
        print(f"{label}: {count}")

    shap_values_correct = explainer(X_correct)

    print("shap_values_correct[0].values.shape")
    print(shap_values_correct[0].values.shape)

    print(f"Computed SHAP values for {len(correct_indices)} correctly predicted samples.")

    all_dfs = []
    for i, idx in enumerate(correct_indices):

        pred_class = y_pred[idx]
        shap_vals = shap_values_correct[i].values[:, pred_class]

        if shap_vals.ndim == 2:
            pred_class = y_pred[idx]
            shap_vals = shap_vals[pred_class]

        assert len(shap_vals) == len(feature_names), \
            f"SHAP value length {len(shap_vals)} doesn't match feature count {len(feature_names)}"

    shap_array = np.stack([sv.values for sv in shap_values_correct])
    abs_shap = np.abs(shap_array)
    mean_shap_per_class = abs_shap.mean(axis=0)
    normalized = mean_shap_per_class / mean_shap_per_class.sum(axis=0, keepdims=True)
    top_k = 10
    top_features_per_class = {}

    for class_idx in range(normalized.shape[1]):
        top_k_indices = np.argsort(normalized[:, class_idx])[::-1][:top_k]
        top_features_per_class[class_idx] = [(feature_names[i], normalized[i, class_idx])
                                         for i in top_k_indices]
        
    output_file = os.path.join(results_dir, f"top_{top_k}_features_per_class.csv")

    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'Feature', 'Mean_SHAP'])

        for class_id, features in top_features_per_class.items():
            class_name = le.inverse_transform([class_id])[0]
            for fname, score in features:
                writer.writerow([class_name, fname, round(score, 6)])

    print(f"Top {top_k} features per class saved to {output_file}")


    return shap_values_correct, correct_indices, explainer

def shap_explain_positive(clf, model_clf, X_test, y_test, feature_names, le):
    print(f"Explaining model predictions using SHAP for model {clf}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("Explaining model predictions using SHAP for all correctly predicted samples")

    explainer = shap.Explainer(model_clf, X_test)
    y_pred = model_clf.predict(X_test)

    correct_indices = np.where(y_pred == y_test)[0]
    if len(correct_indices) == 0:
        raise ValueError("No correctly predicted samples found in test set.")

    X_correct = X_test[correct_indices]
    correct_labels = y_test[correct_indices]

    print("Shape of correct_labels:", correct_labels.shape)

    num_classes = len(le.classes_)
    count_class = {i: (correct_labels == i).sum() for i in range(num_classes)}

    print("Counts of correctly predicted labels per class:")
    for label, count in count_class.items():
        print(f"{label}: {count}")

    shap_values_correct = explainer(X_correct)

    print(f"shap_values_correct type: {type(shap_values_correct)}")
    print(f"shap_values_correct[0] type: {type(shap_values_correct[0])}")
    print(f"shap_values_correct.shape: {getattr(shap_values_correct, 'shape', 'No shape attribute')}")
    print(f"SHAP values tensor shape: {shap_values_correct.values.shape}")
    print(f"Computed SHAP values for {len(correct_indices)} correctly predicted samples.")

    shap_matrix = []

    for i, idx in enumerate(correct_indices):
        pred_class = y_pred[idx]
        shap_array = shap_values_correct[i].values

        # Safe handling of SHAP output shape
        if shap_array.ndim == 2:
            shap_vals = shap_array[:, pred_class]
        elif shap_array.ndim == 1:
            shap_vals = shap_array
        else:
            raise ValueError(f"Unexpected SHAP value shape: {shap_array.shape}")

        shap_matrix.append(shap_vals)

        if i < 3:
            print(f"Sample {i}: SHAP values shape = {shap_vals.shape} for predicted class {pred_class}")

        assert len(shap_vals) == len(feature_names), \
            f"SHAP value length {len(shap_vals)} doesn't match feature count {len(feature_names)}"

    print("All SHAP values have the correct shape.")
    print(f"Last sample SHAP values shape: {shap_vals.shape}")
    shap_matrix = np.array(shap_matrix)
    print(f"Final SHAP matrix shape: {shap_matrix.shape}")

    num_classes_check = shap_values_correct.values.shape[-1] if shap_values_correct.values.ndim == 3 else 1
    K = 15
    csv_path = os.path.join(results_dir, f"{clf}_top_bottom15_genes_all_classes_from_shap_matrix.csv")

    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['class_name', 'gene', 'shap_value', 'rank_type'])

        for cls in range(num_classes_check):
            class_indices = [
                i for i, idx in enumerate(correct_indices)
                if y_pred[idx] == cls and y_test[idx] == cls
            ]

            if not class_indices:
                print(f"Class {cls}: No correctly predicted samples.")
                continue

            shap_cls = shap_matrix[class_indices, :]
            mean_shap_cls = np.mean(shap_cls, axis=0)

            top_idx = np.argsort(-mean_shap_cls)[:K]
            bottom_idx = np.argsort(mean_shap_cls)[:K]

            top_features = [feature_names[i] for i in top_idx]
            top_values = mean_shap_cls[top_idx]

            bottom_features = [feature_names[i] for i in bottom_idx]
            bottom_values = mean_shap_cls[bottom_idx]

            class_name = le.inverse_transform([cls])[0]

            for gene, val in zip(top_features, top_values):
                writer.writerow([class_name, gene, f"{val:.4f}", 'top'])

            for gene, val in zip(bottom_features, bottom_values):
                writer.writerow([class_name, gene, f"{val:.4f}", 'bottom'])

    print(f"Saved top and bottom {K} genes for all classes to {csv_path}")
 

def lime_explain_sample(sample, model, explainer, feature_names, pred_class):
    explanation = explainer.explain_instance(
        sample,
        model.predict_proba,
        num_features=len(feature_names),
        labels=[pred_class]
    )
    weights = dict(explanation.as_list(label=pred_class))
    lime_vector = np.array([weights.get(f, 0.0) for f in feature_names])
    return lime_vector


def lime_explain_positive_parallel(clf, model_clf, X_test, y_test, feature_names, le, n_jobs=4):
    print(f"Explaining model predictions using LIME (parallel) for model {clf}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    y_pred = model_clf.predict(X_test)
    correct_indices = np.where(y_pred == y_test)[0]
    if len(correct_indices) == 0:
        raise ValueError("No correctly predicted samples found in test set.")

    X_correct = X_test[correct_indices]
    correct_labels = y_test[correct_indices]

    explainer = LimeTabularExplainer(
        X_test,
        feature_names=feature_names,
        class_names=le.classes_,
        discretize_continuous=True,
        mode='classification'
    )

    print("Running LIME explanations in parallel...")
    lime_matrix = Parallel(n_jobs=n_jobs)(
        delayed(lime_explain_sample)(
            X_test[idx],
            model_clf,
            explainer,
            feature_names,
            int(y_pred[idx])
        ) for idx in tqdm(correct_indices, desc=f"LIME explaining {clf}")
    )

    lime_matrix = np.array(lime_matrix)
    print(f"LIME matrix shape: {lime_matrix.shape}")

    K = 15
    csv_path = os.path.join(results_dir, f"{clf}_top_bottom15_genes_all_classes_from_lime_matrix.csv")
    num_classes = len(le.classes_)

    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['class_name', 'gene', 'lime_value', 'rank_type'])

        for cls in range(num_classes):
            class_indices = [
                i for i, idx in enumerate(correct_indices)
                if y_pred[idx] == cls and y_test[idx] == cls
            ]
            if not class_indices:
                print(f"Class {cls}: No correctly predicted samples.")
                continue

            lime_cls = lime_matrix[class_indices, :]
            mean_lime_cls = np.mean(lime_cls, axis=0)

            top_idx = np.argsort(-mean_lime_cls)[:K]
            bottom_idx = np.argsort(mean_lime_cls)[:K]

            class_name = le.inverse_transform([cls])[0]

            for i in top_idx:
                writer.writerow([class_name, feature_names[i], f"{mean_lime_cls[i]:.4f}", 'top'])

            for i in bottom_idx:
                writer.writerow([class_name, feature_names[i], f"{mean_lime_cls[i]:.4f}", 'bottom'])

    print(f"Saved top and bottom {K} genes for all classes to {csv_path}")

cmdline_parser = argparse.ArgumentParser('Training')

cmdline_parser.add_argument('-x', '--xai',
                                default="shap",
                                help='model name',
                                type=str)
models = ['lr']
args, unknowns = cmdline_parser.parse_known_args()
adata = ad.read_h5ad('/data1/data/corpus/scDATA/Zheng68K.h5ad')  
X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
feature_names = adata.var_names
le = LabelEncoder()
y = le.fit_transform(adata.obs['celltype'])
X_train, y_train, X_test, y_test = h.split_data(X, y)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

for i, clf in enumerate(models):
    print(f"\nRunning model {i + 1}/{len(models)}: {clf}")
    model_path = f"/data1/data/corpus/scMODEL/{clf}_model_Zheng68K.pkl"

    if os.path.exists(model_path):
        print(f"Loading {clf} model...")
        model_clf = joblib.load(model_path)
    else:
        print("Model not found. Training new model.")
        if clf == "rf":
            model_clf = h.train_rf(X_train, y_train)
        elif clf == "xgb":
            model_clf = h.train_xgboost(X_train, y_train)
        elif clf == "lr":
            model_clf = h.train_logistic_regression(X_train, y_train)
        joblib.dump(model_clf, model_path)
        print(f"Model saved to {model_path}")

    if args.xai == "shap":
        print(f"Running {args.xai} for model :{clf}")
        shap_explain_positive(clf, model_clf, X_test, y_test, feature_names, le)
    elif args.xai == "lime":
        print(f"Running {args.xai} for model :{clf}")
        lime_explain_positive_parallel(clf, model_clf, X_test, y_test, feature_names, le)


