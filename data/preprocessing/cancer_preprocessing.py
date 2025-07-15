import pandas as pd
import os
from glob import glob

input_dir = '/data1/data/corpus/scDATA/cancer/data/pc_UterusG_eac'
combined_path = os.path.join(input_dir, 'pc_UterusG_eac_combined.csv')

if os.path.exists(combined_path):
    os.remove(combined_path)
    print(f"Deleted existing file: {combined_path}\n")

csv_files = glob(os.path.join(input_dir, '*.csv'))

reference_columns = None
mismatched_files = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    file_name = os.path.basename(file_path)
    print(f"File: {file_name}")
    print(f"Shape: {df.shape}")
    #print(f"Columns: {list(df.columns)}\n")

    if reference_columns is None:
        reference_columns = list(df.columns)
    elif list(df.columns) != reference_columns:
        mismatched_files.append((file_name, list(df.columns)))

if mismatched_files:
    print("Mismatched column files:")
    for fname, cols in mismatched_files:
        print(f"- {fname} columns: {cols}")
else:
    print("All files have matching columns.")
