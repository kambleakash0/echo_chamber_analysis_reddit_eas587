import pandas as pd
from pathlib import Path
import os

# --- Configuration ---
# Input directories to scan
RAW_DATA_DIR = 'data'
CLEANED_DATA_DIR = 'data_cleaned'

# Output directories for the samples
JSONL_SAMPLE_DIR = Path('data_sample/original_jsonl')
CSV_SAMPLE_DIR = Path('data_sample/cleaned_csv')

# Number of records to keep from the top of each file
RECORDS_TO_KEEP = 5

# --- Main Script ---

def create_organized_data_samples():
    """
    Scans raw and cleaned data directories, takes the first 5 records from each file,
    and saves them to organized subdirectories within a 'data_sample' folder.
    """
    print("="*60)
    print("Creating Organized Data Samples...")
    print(f"JSONL samples will be saved to: '{JSONL_SAMPLE_DIR}'")
    print(f"CSV samples will be saved to:   '{CSV_SAMPLE_DIR}'")
    print(f"Records to keep per file: {RECORDS_TO_KEEP}")
    print("="*60)

    # 1. Create the output directories if they don't exist
    try:
        os.makedirs(JSONL_SAMPLE_DIR, exist_ok=True)
        os.makedirs(CSV_SAMPLE_DIR, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directories. {e}")
        return

    files_processed = 0
    files_skipped = 0

    # 2. Process the raw JSONL files
    raw_path = Path(RAW_DATA_DIR)
    if not raw_path.exists():
        print(f"\nWarning: Raw data directory '{RAW_DATA_DIR}' not found. Skipping.")
    else:
        print(f"\nScanning directory: '{RAW_DATA_DIR}'...")
        for file_path in sorted(raw_path.glob('*.jsonl')):
            output_file_path = JSONL_SAMPLE_DIR / file_path.name
            try:
                with open(file_path, 'r', encoding='utf-8') as infile, \
                     open(output_file_path, 'w', encoding='utf-8') as outfile:
                    
                    for i, line in enumerate(infile):
                        if i >= RECORDS_TO_KEEP:
                            break
                        outfile.write(line)
                print(f"  ✓ Sampled {RECORDS_TO_KEEP} lines from '{file_path}'")
                files_processed += 1
            except Exception as e:
                print(f"  ✗ Error processing '{file_path.name}': {e}")

    # 3. Process the cleaned CSV files
    cleaned_path = Path(CLEANED_DATA_DIR)
    if not cleaned_path.exists():
        print(f"\nWarning: Cleaned data directory '{CLEANED_DATA_DIR}' not found. Skipping.")
    else:
        print(f"\nScanning directory: '{CLEANED_DATA_DIR}'...")
        for file_path in sorted(cleaned_path.glob('*.csv')):
            output_file_path = CSV_SAMPLE_DIR / file_path.name
            try:
                df_sample = pd.read_csv(file_path, nrows=RECORDS_TO_KEEP)
                df_sample.to_csv(output_file_path, index=False, encoding='utf-8')
                print(f"  ✓ Sampled {RECORDS_TO_KEEP} rows from '{file_path}'")
                files_processed += 1
            except Exception as e:
                print(f"  ✗ Error processing '{file_path.name}': {e}")

    print("\n" + "="*60)
    print("SAMPLING SUMMARY:")
    print(f"  Successfully processed: {files_processed} files")
    print(f"  Sample data saved in '{JSONL_SAMPLE_DIR.parent}/'")
    print("\nProcess completed!")

if __name__ == "__main__":
    create_organized_data_samples()