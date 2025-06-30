import scanpy as sc
import pandas as pd
import numpy as np
from get_embedding import main_gene_selection
import os

csv_path = '/data1/data/corpus/scDATA/Zheng68K_scf_preprocessed.csv'

if not os.path.exists(csv_path):
    adata = sc.read_h5ad('/data1/data/corpus/scDATA/Zheng68K.h5ad')
    X_df = pd.DataFrame(adata.X.toarray(), columns=adata.var_names, index=adata.obs_names)
    X_df['label'] = adata.obs['celltype'].values
    gene_list_df = pd.read_csv('OS_scRNA_gene_index.19264.tsv', delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])
    X_df, to_fill_columns, var = main_gene_selection(X_df, gene_list + ['label'])
    X_df.to_csv(csv_path)
    df = pd.read_csv(csv_path, nrows=1)
    column_names = df.columns.tolist()
else:
    df = pd.read_csv(csv_path, nrows=1)
    column_names = df.columns.tolist()

print(column_names)
