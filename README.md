# scRNAseq

This repository provides scripts, models, and utilities for **cell type classification** and experiments on single-cell RNA sequencing (scRNA-seq) data.  
It covers baseline machine learning models, deep neural networks, explainability methods, and foundation models (e.g., scBERT, scFoundation).

---

## Project Structure

scRNAseq/  
│  
├── cell_type_classification/   # Main pipeline for cell classification using baseline classifiers and XAI (ML, NN, GRU, XAI)    
│  
├── data/                       # pre-processing dataset  
│  
├── initial_version/            # earlier versions made for testing setup 
│  
├── scbert/                     # Experiments with scBERT (e.g., finetuning/ kidney classification/ XAI)  
│  
├── scfoundation/               # Experiments with scFoundation (e.g., finetuning/ kidney classification/ XAI)   

---

### 4. Foundation Model Experiments

- **scBERT (`scbert/`)**  
  Fine-tuning of scBERT on cell type and kidney cancer scRNA-seq datasets for classification.  

- **scFoundation (`scfoundation/`)**  
  Experiments with scFoundation models, including cancer datasets.  

These directories contain additional scripts for interpreabillity for these foundation models.

---

### 5. Results

- All experiment outputs (metrics, checkpoints, logs, feature importance, etc.) are stored in `cell_type_classification/results/`.  
- Results from foundation models (scBERT and scFoundation) are stored in their respective folders.

---

## Requirements

- Python ≥ 3.9  
- Poetry ≥ 1.4  
- PyTorch  
- scikit-learn  
- numpy, pandas, matplotlib  
- SHAP (for interpretability)  
- ...
