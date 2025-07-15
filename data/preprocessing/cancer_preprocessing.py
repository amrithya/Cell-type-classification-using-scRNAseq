import pandas as pd
import os
from glob import glob

input_dir = '/data1/data/corpus/scDATA/cancer/data/pc_UterusG_eac'
csv_files = glob(os.path.join(input_dir, '*.csv'))

all_dfs = []
reference_columns = None
skipped_files = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    if reference_columns is None:
        reference_columns = list(df.columns)
    elif list(df.columns) != reference_columns:
        print(f"Skipping {file_path} due to column mismatch.")
        skipped_files.append(file_path)
        continue
    class_name = os.path.splitext(os.path.basename(file_path))[0]
    df['class'] = class_name
    all_dfs.append(df)

if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(os.path.join(input_dir, 'pc_UterusG_eac_combined.csv'), index=False)
    print(f"Combined CSV saved. Skipped {len(skipped_files)} files.")
else:
    print("No valid files to combine.")
