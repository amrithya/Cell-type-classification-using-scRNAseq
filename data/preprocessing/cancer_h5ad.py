import scanpy as sc

adata_path = '/data1/data/corpus/scDATA/cancer/data/pc_UterusG_eac/pc_UterusG_eac.h5ad'
adata = sc.read_h5ad(adata_path)

print(adata)
print("\nObservation columns:")
print(adata.obs.head())

print("\nVariable (gene) names:")
print(adata.var_names[:10])

print("\nExpression matrix shape:")
print(adata.X.shape)

print("\nClass counts:")
print(adata.obs['class'].value_counts())
