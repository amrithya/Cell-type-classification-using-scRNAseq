import scanpy as sc
import pandas as pd
import numpy as np
from get_embedding import main_gene_selection
import os

csv_path = '/data1/data/corpus/scDATA/cancer/data/mt_kidney_rcc_cca_paa_2/grade/mt_kidney_rcc_cca_paa_2_scf_preprocessed.csv'

if not os.path.exists(csv_path):
    adata = sc.read_h5ad('/data1/data/corpus/scDATA/cancer/data/mt_kidney_rcc_cca_paa_2/grade/mt_kidney_rcc_cca_paa_2.h5ad')
    X_df = pd.DataFrame(adata.X.toarray(), columns=adata.var_names, index=adata.obs_names)
    X_df['label'] = adata.obs['class'].values
    gene_list_df = pd.read_csv('OS_scRNA_gene_index.19264.tsv', delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])
    X_df, to_fill_columns, var = main_gene_selection(X_df, gene_list + ['label'])
    X_df.to_csv(csv_path)
    df = pd.read_csv(csv_path, nrows=1)
    column_names = df.columns.tolist()
else:
    df = pd.read_csv(csv_path, nrows=1)
    column_names = df.columns.tolist()

    embedding_path = "/data1/data/corpus/scDATA/scfoundation/mt_kidney_rcc_cca_paa_2_scf.npy"
    embeddings = np.load(embedding_path)

    print("Embedding shape:", embeddings.shape)
    print("Sample values:\n", embeddings[:2])


