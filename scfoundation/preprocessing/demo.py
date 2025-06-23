# -*- coding: utf-8 -*-

from scRNA_workflow import *
sc.settings.figdir='./figures_new/'

path = '/data1/data/corpus/scDATA/Zheng68K.h5ad'
adata = sc.read_10x_mtx(path)

X_df= pd.DataFrame(sparse.csr_matrix.toarray(adata.X),index=adata.obs.index.tolist(),columns=adata.var.index.tolist())
gene_list_df = pd.read_csv('./OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
gene_list = list(gene_list_df['gene_name'])
X_df, to_fill_columns, var = main_gene_selection(X_df, gene_list)
adata_uni = sc.AnnData(X_df)
adata_uni.obs = adata.obs
adata_uni.uns = adata.uns

adata_uni = BasicFilter(adata_uni,qc_min_genes=200,qc_min_cells=0) # filter cell and gene by lower limit
adata_uni = QC_Metrics_info(adata_uni)

save_path = '/data1/data/corpus/scDATA/scfoundation/demo.h5ad'
save_adata_h5ad(adata_uni,save_path)