import pandas as pd
from pathlib import Path
import os

# Configuration
data_dir = 'data'
output_dir = 'data_cleaned'

# --- Define the master column lists ---

# Columns to keep for POSTS files
POST_COLUMNS_TO_KEEP = [
    'id', 'subreddit', 'created_utc', 'title', 'selftext', 'url',
    'domain', 'is_self', 'score', 'num_comments', 'upvote_ratio',
    'author', 'stickied', 'over_18', 'permalink'
]

# Columns to keep for COMMENTS files
COMMENT_COLUMNS_TO_KEEP = [
    'id', 'subreddit', 'created_utc', 'body', 'link_id', 'parent_id',
    'score', 'controversiality', 'author', 'is_submitter'
]

# --- Main script ---

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Starting standardization process...")
print(f"Input directory: '{data_dir}'")
print(f"Output directory: '{output_dir}'")
print("=" * 60)

# Get all CSV files from the data directory
csv_files = list(Path(data_dir).glob('*.csv'))

if not csv_files:
    print("No CSV files found to process.")
    exit()

processed_files = 0
total_rows_in = 0
total_rows_out = 0

for csv_file in csv_files:
    print(f"\nProcessing: {csv_file.name}")
    try:
        # Determine if it's a posts or comments file
        if 'posts' in csv_file.name.lower():
            columns_to_keep = POST_COLUMNS_TO_KEEP
            file_type = "Posts"
        elif 'comments' in csv_file.name.lower():
            columns_to_keep = COMMENT_COLUMNS_TO_KEEP
            file_type = "Comments"
        else:
            print(f"  - Skipping file: does not contain 'posts' or 'comments' in name.")
            continue

        # Load the CSV
        df = pd.read_csv(csv_file, low_memory=False)
        original_rows = len(df)
        original_cols = len(df.columns)
        total_rows_in += original_rows

        # Find which columns are actually present in the file
        # This prevents errors if a column is missing from one file
        final_columns = [col for col in columns_to_keep if col in df.columns]
        
        missing_cols = set(columns_to_keep) - set(final_columns)
        if missing_cols:
            print(f"  - Warning: The following columns were not found and will be skipped: {list(missing_cols)}")

        # Select and reorder the columns
        df_cleaned = df[final_columns]

        # Define output path
        output_path = Path(output_dir) / csv_file.name

        # Save the cleaned dataframe
        df_cleaned.to_csv(output_path, index=False, encoding='utf-8')

        cleaned_rows = len(df_cleaned)
        total_rows_out += cleaned_rows

        print(f"  - Type: {file_type}")
        print(f"  - Columns reduced from {original_cols} to {len(df_cleaned.columns)}")
        print(f"  - Rows: {cleaned_rows:,}")
        print(f"  ✓ Saved cleaned file to: {output_path}")
        processed_files += 1

    except Exception as e:
        print(f"  ✗ Error processing {csv_file.name}: {e}")

print("\n" + "=" * 60)
print("STANDARDIZATION SUMMARY:")
print(f"  Files processed: {processed_files}/{len(csv_files)}")
print(f"  Total rows in: {total_rows_in:,}")
print(f"  Total rows out: {total_rows_out:,}")
print("\nProcess completed!")