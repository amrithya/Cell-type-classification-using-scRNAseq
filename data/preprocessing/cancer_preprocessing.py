import pandas as pd
import os
from glob import glob

input_dir = '/data1/data/corpus/scDATA/cancer/data/pc_UterusG_eac'
csv_files = glob(os.path.join(input_dir, '*.csv'))

column_sets = {}
all_equal = True

for file_path in csv_files:
    df = pd.read_csv(file_path, nrows=1)
    file_name = os.path.basename(file_path)
    columns = list(df.columns)
    column_sets[file_name] = columns

reference_file, reference_columns = next(iter(column_sets.items()))

for fname, cols in column_sets.items():
    if cols != reference_columns:
        all_equal = False
        print(f"{fname} has different columns.")
        missing = set(reference_columns) - set(cols)
        extra = set(cols) - set(reference_columns)
        if missing:
            print(f"  Missing: {missing}")
        if extra:
            print(f"  Extra: {extra}")
        print()
        
if all_equal:
    print("All CSV files have identical feature (column) names.")
else:
    print("Some files have mismatched columns.")
