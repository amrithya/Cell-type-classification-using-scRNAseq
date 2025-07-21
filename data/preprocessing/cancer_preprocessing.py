import pandas as pd
import os
from glob import glob
import scanpy as sc
import scipy.sparse as sp

input_dir = '/data1/data/corpus/scDATA/cancer/data/mt_kidney_rcc_cca_paa_2/grade'
combined_path = os.path.join(input_dir, 'mt_kidney_rcc_cca_paa_2_combined.csv')
output_h5ad = os.path.join(input_dir, 'mt_kidney_rcc_cca_paa_2.h5ad')

if os.path.exists(combined_path):
    df = pd.read_csv(combined_path)
    print(f"mt_kidney_rcc_cca_paa_2_combined.csv shape: {df.shape}")
else:
    csv_files = glob(os.path.join(input_dir, '*.csv'))
    all_dfs = []

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        file_name = os.path.basename(file_path)
        print(f"{file_name} shape: {df.shape}")
        df['class'] = os.path.splitext(file_name)[0]
        all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)
    print("\nCombined DataFrame preview:")
    print(df.head())
    df.to_csv(combined_path, index=False)
    print(f"\nCombined CSV saved to {combined_path}")

df = pd.read_csv(combined_path)

df = df.set_index('gene_id')
X = df.drop(columns=['class'])
obs = pd.DataFrame({'class': df['class']})
obs.index = df.index
var = pd.DataFrame(index=X.columns)
X_sparse = sp.csr_matrix(X.values)

adata = sc.AnnData(X=X_sparse, obs=obs, var=var)
adata.write(output_h5ad, compression='gzip')
print(f"Saved: {output_h5ad}")
