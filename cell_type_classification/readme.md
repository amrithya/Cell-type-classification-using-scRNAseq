# Cell Type Classification

This repository contains scripts and utilities for training, evaluating, and interpreting machine learning models for cell type classification using gene expression data.

## Project Structure

```
cell_type_classification/
│
├── poetry_demo/              # Example or test project managed by Poetry
├── results/                  # Output directory for trained models, logs, and metrics
├── sc-foundation-lr/         # Replicated LR from sc-foundation-eval repo 
├── tests/                    # Tests for your setup
│
├── gene_final.py             # Finalized pipeline for cell classification with basline ML
├── helper.py                 # Utility functions
├── nn_model.py               # Neural network model definitions
├── preprocess.py             # Preprocessing pipeline for scRNA-seq data
├── top_genes.py              # Extracts feature importance ( not used as a main result)
├── xai_model.py              # Explainability/interpretability pipeline (e.g., SHAP)
│
├── run_gru.sh                # Run script for GRU-based models
├── run_ml.sh                 # Run script for baseline ML models (e.g., logistic regression, RF)
├── run_nn.sh                 # Run script for neural network training
├── run_top_genes.sh          # Run script to extract top influential genes
├── run_xai.sh                # Run explainability pipeline
│
├── pyproject.toml            # Poetry project configuration
├── poetry.lock               # Poetry dependency lock file
├── README.rst                # Legacy readme
└── .gitignore                # Git ignore file
```

## Installation

This project uses Poetry for dependency management.

```bash
# Install dependencies
poetry install
```

## Usage

### Preprocessing

```bash
poetry run python preprocess.py
```

### Train Neural Network

```bash
bash run_nn.sh
```

### Train GRU Model

```bash
bash run_gru.sh
```

### Baseline ML Models

```bash
bash run_ml.sh
```

### Extract Top Genes

```bash
bash run_top_genes.sh
```

### Explainability (XAI)

```bash
bash run_xai.sh
```


## Results

All experiment outputs (metrics, checkpoints, gene importance, etc.) are saved in the `results/` directory.

## Requirements

* Python ≥ 3.9
* Poetry ≥ 1.4
* PyTorch, scikit-learn, numpy, pandas, matplotlib
* SHAP (for explainability)

(See `pyproject.toml` for the full list of dependencies.)
