import scanpy as sc

adata = sc.read_h5ad("/data1/data/corpus/scDATA/Zheng68K.h5ad")
print(adata.X[:5])
print("Data loaded successfully from /data1/data/corpus/scDATA/Zheng68K.h5ad")
print(f"Data shape: {adata.shape}")
print(f"Number of observations: {adata.n_obs}, Number of variables: {adata.n_vars}")
print(f"Number of observations: {adata.n_obs}, Number of variables: {adata.n_vars}")
print("Variable names:", adata.var_names[:10])
print("Observation names:", adata.obs_names[:10])
print("First 5 observations:", adata.obs.head())
print("First 5 variables:", adata.var.head())
