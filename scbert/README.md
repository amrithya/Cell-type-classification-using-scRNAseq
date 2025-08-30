# scBERT

[![python >3.6.8](https://img.shields.io/badge/python-3.6.8-brightgreen)](https://www.python.org/) 

### scBERT as a Large-scale Pretrained Deep Language Model for Cell Type Annotation of Single-cell RNA-seq Data
Reliable cell type annotation is a prerequisite for downstream analysis of single-cell RNA sequencing data. Existing annotation algorithms typically suffer from improper handling of batch effect, lack of curated marker gene lists, or difficulty in leveraging the latent gene-gene interaction information. Inspired by large scale pretrained langurage models, we present a pretrained deep neural network-based model scBERT (single-cell Bidirectional Encoder Representations from Transformers) to overcome the above challenges. scBERT follows the state-of-the-art paradigm of pre-train and fine-tune in the deep learning field. In the first phase of scBERT, it obtains a general understanding of gene-gene interaction by being pre-trained on huge amounts of unlabeled scRNA-seq data. The pre-trained scBERT can then be used for the cell annotation task of unseen and user-specific scRNA-seq data through supervised fine-tuning. For more information, please refer to [https://www.biorxiv.org/content/10.1101/2021.12.05.471261v1](https://www.biorxiv.org/content/10.1101/2021.12.05.471261v1)

# Install

[![scipy-1.5.4](https://img.shields.io/badge/scipy-1.5.4-yellowgreen)](https://github.com/scipy/scipy) [![torch-1.8.1](https://img.shields.io/badge/torch-1.8.1-orange)](https://github.com/pytorch/pytorch) [![numpy-1.19.2](https://img.shields.io/badge/numpy-1.19.2-red)](https://github.com/numpy/numpy) [![pandas-1.1.5](https://img.shields.io/badge/pandas-1.1.5-lightgrey)](https://github.com/pandas-dev/pandas) [![scanpy-1.7.2](https://img.shields.io/badge/scanpy-1.7.2-blue)](https://github.com/theislab/scanpy) [![scikit__learn-0.24.2](https://img.shields.io/badge/scikit__learn-0.24.2-green)](https://github.com/scikit-learn/scikit-learn) [![transformers-4.6.1](https://img.shields.io/badge/transformers-4.6.1-yellow)](https://github.com/huggingface/transformers)

# Data

The data can be downloaded from these links. If you have any question, please contact fionafyang@tencent.com.
 
https://drive.weixin.qq.com/s?k=AJEAIQdfAAozQt5B8k
https://drive.google.com/file/d/1fNZbKx6LPeoS0hbVYJFI8jlDlNctZxlU/view?usp=sharing

# Checkpoint 

The pre-trained model checkpoint can be downloaded from this link. If you have any question, please contact fionafyang@tencent.com.

https://drive.weixin.qq.com/s?k=AJEAIQdfAAoUxhXE7r

# scBERT Experiments

This folder contains scripts, utilities, and run pipelines for **fine-tuning scBERT** on scRNA-seq datasets (e.g., zheng and cancer).  
It also includes scripts for **interpretability methods** such as Integrated Gradients, DeepLIFT, Gradient-based methods, SHAP, and Attention-based explanations.

---

## Folder Structure

scbert/  
│  
├── IG_outputs/                 # Results from Integrated Gradients runs  
├── archive/                    # Archived logs or older runs  
├── ckpts/                      # Saved checkpoints of fine-tuned scBERT models  
├── deeplift_outputs/           # Results from DeepLIFT interpretability runs  
├── performer_pytorch/          # Performer-based transformer variant implementation  
├── poetry_demo/                # Poetry demo environment  
├── results/                    # Final metrics, outputs, logs  
│  
├── LICENSE                     # License file  
├── README.md                   # Documentation for this folder  

---

## Key Scripts

- **Fine-tuning**
  - `finetune.py` – Base script for fine-tuning scBERT  
  - `finetune_cancer.py` – Fine-tuning on cancer datasets  
  - `finetune_lr.py` – Logistic regression head with scBERT embeddings  
  - `besteffort_finetuning.py` – Optimized fine-tuning strategy  

- **Interpretability**
  - `finetune_IG.py` – Integrated Gradients  
  - `finetune_deeplift.py` – DeepLIFT  
  - `finetune_gradcam.py` – Gradient-based CAM  
  - `finetune_shap.py` – SHAP  
  - `finetune_attn.py` – Attention-based explanations  

- **Run scripts**
  - `run_finetune.sh` – Standard fine-tuning run  
  - `run_finetune_IG.sh` – Run fine-tuning with Integrated Gradients  
  - `run_finetune_deeplift.sh` – Run with DeepLIFT  
  - `run_finetune_gradcam.sh` – Run with Grad-CAM  
  - `run_finetune_shap.sh` – Run with SHAP  
  - `run_finetune_attn.sh` – Run with Attention explanations  
  - `run_cancer_preprocess.sh` – Preprocessing step for cancer datasets  

- **Utilities**
  - `utils.py` – Helper functions for preprocessing, training, or logging  

---

