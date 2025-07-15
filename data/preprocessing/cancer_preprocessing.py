import pandas as pd
import os
from glob import glob

input_dir = '/data1/data/corpus/scDATA/cancer/data/pc_UterusG_eac'
csv_files = glob(os.path.join(input_dir, '*.csv'))

for file_path in csv_files:
    df = pd.read_csv(file_path)
    file_name = os.path.basename(file_path)
    print(f"File: {file_name}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
