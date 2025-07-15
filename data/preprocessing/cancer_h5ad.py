import scanpy as sc
import scipy.sparse as sp
import numpy as np

adata_path = '/data1/data/corpus/scDATA/cancer/data/pc_UterusG_eac/pc_UterusG_eac.h5ad'
adata = sc.read_h5ad(adata_path)

print(adata)
print("\nObservation columns:")
print(adata.obs.head())

print("\nVariable (gene) names:")
print(adata.var_names[:10])

print("\nExpression matrix shape:")
print(adata.X.shape)

print("\nIs expression matrix sparse?", sp.issparse(adata.X))

print("\nClass counts:")
print(adata.obs['class'].value_counts())

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

print("\nAfter normalization and log1p, expression matrix shape:", adata.X.shape)

print("\nExpression matrix head (first 5 cells × first 5 genes):")
if sp.issparse(adata.X):
    print(adata.X[:5, :5].toarray())
else:
    print(adata.X[:5, :5])

adata.write(adata_path)
print(f"Saved normalized and log1p transformed data to {adata_path}")
