import pandas as pd
from pathlib import Path

# Configuration
data_dir = 'data'

# Function to display file information
def analyze_csv_files():
    csv_files = sorted(Path(data_dir).glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in '{data_dir}' directory!")
        return
    
    print(f"📊 CSV FILES ANALYSIS ({len(csv_files)} files found)")
    print("=" * 100)
    
    total_size = 0
    total_rows = 0
    
    for i, file in enumerate(csv_files, 1):
        try:
            # Get basic info without loading full data
            df_info = pd.read_csv(file, nrows=0)
            file_size = file.stat().st_size / (1024 * 1024)  # MB
            
            # Count rows efficiently
            with open(file, 'r') as f:
                row_count = sum(1 for _ in f) - 1  # excluding header
            
            total_size += file_size
            total_rows += row_count
            
            print(f"\n{i:2d}. 📄 {file.name}")
            print(f"     📏 Size: {file_size:.2f} MB | 📊 Rows: {row_count:,} | 📋 Columns: {len(df_info.columns)}")
            
            # Display columns in groups of 4
            columns = df_info.columns.tolist()
            print("     🏷️  Columns:")
            for j in range(0, len(columns), 4):
                col_group = columns[j:j+4]
                print(f"         {' | '.join(col_group)}")
                
        except Exception as e:
            print(f"\n{i:2d}. ❌ {file.name} - Error: {e}")
    
    print("\n" + "=" * 100)
    print(f"📈 SUMMARY: {len(csv_files)} files | {total_size:.2f} MB total | {total_rows:,} total rows")

# Run the analysis
analyze_csv_files()