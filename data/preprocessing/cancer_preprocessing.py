import pandas as pd
import os
from glob import glob

input_dir = '/data1/data/corpus/scDATA/cancer/data/pc_UterusG_eac'
csv_files = glob(os.path.join(input_dir, '*.csv'))

all_dfs = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    file_name = os.path.basename(file_path)
    print(f"{file_name} shape: {df.shape}")
    
    df['class'] = os.path.splitext(file_name)[0]
    all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)
print("\nCombined DataFrame preview:")
print(combined_df.head())

combined_path = os.path.join(input_dir, 'pc_UterusG_eac_combined.csv')
combined_df.to_csv(combined_path, index=False)
print(f"\nCombined CSV saved to {combined_path}")
