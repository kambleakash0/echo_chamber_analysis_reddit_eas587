import pandas as pd
import json
import os
from pathlib import Path

# Configuration
data_dir = 'data'
output_dir = data_dir  # Change this if you want CSV files in a different directory

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Check if data directory exists
if not os.path.exists(data_dir):
    print(f"Error: Directory '{data_dir}' does not exist!")
    print("Please create the 'data' directory and place your JSONL files there.")
    exit(1)

# Get all JSONL files from the data directory
jsonl_files = list(Path(data_dir).glob('*.jsonl'))

if not jsonl_files:
    print(f"No JSONL files found in '{data_dir}' directory!")
    print("Please make sure your files have .jsonl extension.")
    exit(1)

print(f"Found {len(jsonl_files)} JSONL files to process:")
for file in jsonl_files:
    print(f"  - {file.name}")

print(f"\nOutput directory: {output_dir}")
print("\n" + "="*60)

# Process each JSONL file
successful_conversions = 0
total_records = 0

for jsonl_file in jsonl_files:
    print(f"\nProcessing: {jsonl_file.name}")

    try:
        # Read JSONL file (each line is a separate JSON object)
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        json_obj = json.loads(line)
                        data.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"  Warning: Skipping invalid JSON on line {line_num}: {e}")

        if not data:
            print(f"  Warning: No valid JSON objects found in {jsonl_file.name}")
            continue

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Generate output filename in the output directory
        output_file = Path(output_dir) / f"{jsonl_file.stem}.csv"

        # Write DataFrame to CSV file
        df.to_csv(output_file, index=False, encoding='utf-8')

        print(f"  ✓ Successfully converted to: {output_file.name}")
        print(f"    Records: {len(df)}, Columns: {len(df.columns)}")

        successful_conversions += 1
        total_records += len(df)

    except Exception as e:
        print(f"  ✗ Error processing {jsonl_file.name}: {e}")

print("\n" + "="*60)
print("CONVERSION SUMMARY:")
print(f"  Files processed: {successful_conversions}/{len(jsonl_files)}")
print(f"  Total records converted: {total_records:,}")
print(f"  Output location: {output_dir}/")
print("\nConversion process completed!")