import pandas as pd
import anndata

csv_path = "IG_outputs/IG_top_bottom_genes_per_class.csv"
h5ad_path = "/data1/data/corpus/scDATA/Zheng68K.h5ad"
output_csv_path = "IG_outputs/IG_top_bottom_genes_per_class_with_genes.csv"

df = pd.read_csv(csv_path)
adata = anndata.read_h5ad(h5ad_path)

class_map = {
    0: 'CD14+ Monocyte',
    1: 'CD19+ B',
    2: 'CD34+',
    3: 'CD4+ T Helper2',
    4: 'CD4+/CD25 T Reg',
    5: 'CD4+/CD45RA+/CD25- Naive T',
    6: 'CD4+/CD45RO+ Memory',
    7: 'CD56+ NK',
    8: 'CD8+ Cytotoxic T',
    9: 'CD8+/CD45RA+ Naive Cytotoxic',
    10: 'Dendritic'
}

df['class'] = df['class'].map(class_map)

df['gene_name'] = df['gene_index'].apply(lambda idx: adata.var_names[int(idx)])

df.to_csv(output_csv_path, index=False)
