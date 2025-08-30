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

## Usage in our project 

