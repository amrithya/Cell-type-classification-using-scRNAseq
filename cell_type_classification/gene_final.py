import os
import torch
import argparse
import numpy as np
import helper as h
import scanpy as sc
import pandas as pd
import anndata as ad
import torch.nn as nn
import nn_model as nnm 
import helper as h
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder


def run_model(model,X_train, y_train, X_test, y_test, samp, le):
    """
    Trains and evaluates a classification model (Logistic Regression or Random Forest)
    on the given AnnData object. Optionally performs downsampling to balance class labels.

    Parameters:
    -----------
    model : str
        The model to train, either "lr" for Logistic Regression or "rf" for Random Forest.

    X_train : np.ndarray
        The training feature matrix.
    y_train : np.ndarray
        The training labels.
    X_test : np.ndarray
        The test feature matrix.
    y_test : np.ndarray
        The test labels.

    Returns:
    --------
    None
    """
    if model == "lr":
        lr = h.train_logistic_regression(X_train, y_train)
        h.evaluate_model(lr, X_train, y_train,le,"train",model,samp)
        h.evaluate_model(lr, X_test, y_test,le,"test",model,samp)
        #shap_values, explainer = xai.shap_explain(lr, X_test, y_test, le)
    elif model == "rf":
        rf = h.train_rf(X_train, y_train)
        h.evaluate_model(rf, X_train, y_train, le,"train",model,samp)
        h.evaluate_model(rf, X_test, y_test, le,"test",model,samp)
        #h.shap_explainer(lr, X_train, X_test)

    elif model == "sgd":
        rf = h.train_sgd(X_train, y_train)
        h.evaluate_model(rf, X_train, y_train, le,"train",model,samp)
        h.evaluate_model(rf, X_test, y_test, le,"test",model,samp)
        #h.shap_explainer(lr, X_train, X_test)
    elif model == "xg":
        xg = h.train_xgboost(X_train, y_train)
        h.evaluate_model(xg, X_train, y_train, le,"train",model,samp)
        h.evaluate_model(xg, X_test, y_test, le,"test",model,samp)
        #h.shap_explainer(lr, X_train, X_test)



if __name__ == "__main__":

    print("Running gene_final")

    """
    Main script for training and evaluating classification models on single-cell gene expression data.

    This script performs the following:
    1. Loads the PBMC68k dataset from an .h5ad file.
    2. Preprocesses the data (normalization, log transformation, label encoding).
    3. Accepts command-line arguments to:
        - Choose the model type (logistic regression or random forest).
        - Enable or disable downsampling for class balancing.
        - Specify the output CSV file to store evaluation results.
    4. Trains the model on the training data.
    5. Evaluates the model on both training and test data.
    6. Saves the performance metrics and confusion matrix to disk.

    Example usage:
        python gene_final.py -m rf -d -c  # if you run on cluster,include -c
        python gene_final.py --model lr --samp
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cmdline_parser = argparse.ArgumentParser('Training')

    cmdline_parser.add_argument('-m', '--model',
                                default="lr",
                                help='model name',
                                type=str)
    
    cmdline_parser.add_argument('-d', '--samp',
                                action='store_true',
                                help='enable sampling')
    
    cmdline_parser.add_argument('-c', '--cluster',
                                action='store_true',
                                help='dataset file location')
    
    cmdline_parser.add_argument('-s', '--smote',
                                action='store_true',
                                help='smote' )
    
    args, unknowns = cmdline_parser.parse_known_args()

    print(f"Loading data => model: {args.model}, samp: {args.samp}, cluster: {args.cluster}" )
    X_train, y_train, X_test, y_test, le, gene_names = h.load_data(args.samp, args.cluster, args.smote)
    print("Data loaded successfully.")
    if args.model == "nn":
        print("Preprocessing data for nn")
        train_data, test_data, weights,le, input_size, output_size  = h.preprocess_data_nn(device, X_train, y_train, X_test, y_test,le)
        hidden_sizes = [128]
        lr_rates = [0.001]
        dropout_rates = [0.2]
        results = []
        for hidden_size in hidden_sizes:
            for dropout in dropout_rates:
                for lr in lr_rates:
                    model, test_accuracy, train_accuracy, test_correct_indices = nnm.train_nn(device, train_data, test_data, lr, weights, input_size, output_size, dropout, hidden_size)
                    results.append((hidden_size, lr, dropout, train_accuracy, test_accuracy))
                    nnm.analyze_gradient_input_classwise(model, X_test, y_test, test_correct_indices, gene_names, le, device)
                    #nnm.shap_explain_nn(model, train_data, test_data, gene_names, le, device=device)
                    #lrp = nnm.LRP(model)
                    #nnm.analyze_lrp_classwise(model, lrp, X_test, y_test, test_correct_indices, gene_names, le, device)
        
        print("\nSummary of Results:")
        for hidden_size, lr, dropout, train_acc, acc in results:
            print(f"Hidden: {hidden_size}, LR: {lr}, Dropout: {dropout} => Train Accuracy: {train_acc:.2f}, Test Accuracy: {acc:.2f}%")

    elif args.model == "nn_gru":
        print("Preprocessing data for nn")
        train_data, test_data, weights,le, input_size, output_size  = h.preprocess_data_nn(device, X_train, y_train, X_test, y_test,le)
        hidden_sizes = [256]
        lr_rates = [0.001]
        dropout_rates = [0.5]
        results = []
        for hidden_size in hidden_sizes:
            for dropout in dropout_rates:
                for lr in lr_rates:
                    model, test_accuracy, train_accuracy, test_correct_indices = nnm.train_gru(device, train_data, test_data, lr, weights, input_size, output_size, dropout, hidden_size)
                    results.append((hidden_size, lr, dropout, train_accuracy, test_accuracy))
                    #nnm.shap_explain_nn(model, test_data, gene_names, le, device=device)
                    #lrp = nnm.LRP(model)
                    #nnm.analyze_lrp_classwise(model, lrp, X_test, y_test, test_correct_indices, gene_names, le, device)
        
        print("\nSummary of Results:")
        for hidden_size, lr, dropout, train_acc, acc in results:
            print(f"Hidden: {hidden_size}, LR: {lr}, Dropout: {dropout} => Train Accuracy: {train_acc:.2f}, Test Accuracy: {acc:.2f}%")

    else:
        models = ["lr", "rf", "sgd", "xg"]
        for model_name in models:
            print(f"Running model: {model_name}")
            run_model(model_name, X_train, y_train, X_test, y_test, args.samp, le)
    

