import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scipy import sparse

panglao = sc.read_h5ad('/data1/data/corpus/scDATA/panglao_human.h5ad')
data = sc.read_h5ad('/data1/data/corpus/scDATA/cancer/data/pc_UterusG_eac/pc_UterusG_eac.h5ad')

counts = sparse.lil_matrix((data.X.shape[0], panglao.X.shape[1]), dtype=np.float32)
ref = panglao.var_names.tolist()
obj = data.var_names.tolist()

for i in range(len(ref)):
    if ref[i] in obj:
        loc = obj.index(ref[i])
        counts[:, i] = data.X[:, loc]

counts = counts.tocsr()
new = ad.AnnData(X=counts)
new.var_names = ref
new.obs_names = data.obs_names
new.obs = data.obs
new.uns = panglao.uns

sc.pp.filter_cells(new, min_genes=200)
new.write('./data/preprocessed_data_pc_UterusG_eac_bert.h5ad')
print(new.shape)