"""
Script to process CAID format files and generate FASTA + annotations CSV.
python -m scripts.process_caid
"""
from pathlib import Path
from src.caid import process_caid_file

DATASET = "disorder_nox"
INPUT_DIR = "data/raw/caid3_3/"
OUTPUT_DIR = "data/caid3_3/"

if __name__ == "__main__":

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    caid_file = input_dir / f"{DATASET}.fasta"
    output_fasta = output_dir / f"{DATASET}.fasta"
    output_csv = output_dir / f"{DATASET}.csv"
    
    num_records, num_annotations = process_caid_file(caid_file, output_fasta, output_csv)
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Records processed: {num_records}")
    print(f"Annotations generated: {num_annotations}")
    print("=" * 60)
