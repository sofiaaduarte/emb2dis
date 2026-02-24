"""
CAID format parser for protein sequences and annotations.
"""
from pathlib import Path
import pandas as pd


def parse_caid_record(acc: str, sequence: str, annotation: str) -> dict:
    """
    Parse a single CAID record (acc, sequence, annotation).

    Args:
        acc: Protein ID
        sequence: Amino acid sequence
        annotation: Annotation string with '1', '0', '-'
    Returns:
        Dictionary with 'id', 'sequence', 'annotations' (list of regions)
        (first amino acid is position 1)
    """
    if len(sequence) != len(annotation):
        raise ValueError(f"Sequence and annotation length mismatch for {acc}")
    
    # Parse annotation into regions
    regions = []
    current_label = None
    start = None
    
    for i, (aa, ann) in enumerate(zip(sequence, annotation)):
        if ann == '-':
            # Unknown position (-) close current region if any
            if current_label is not None:
                regions.append({
                    'start': start,
                    'end': i,
                    'label': '1' if current_label == '1' else '0'
                })
                current_label = None
                start = None
        else:
            # Known position
            if current_label == ann:
                continue # Continue current region
            else:
                # Close previous region and start new one
                if current_label is not None:
                    regions.append({
                        'start': start,
                        'end': i,
                        'label': '1' if current_label == '1' else '0'
                    })
                current_label = ann
                start = i + 1
    
    # Close last region if exists
    if current_label is not None:
        regions.append({
            'start': start,
            'end': len(sequence),
            'label': '1' if current_label == '1' else '0'
        })
    
    return {
        'id': acc,
        'sequence': sequence,
        'annotations': regions
    }


def read_caid_file(caid_file: str) -> list:
    """
    Read a CAID format file and parse all records.
    """
    records = []
    
    with open(caid_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for header line (starts with '>')
        if line.startswith('>'):
            acc = line[1:]  # Remove '>'
            
            # Next line should be sequence
            if i + 1 >= len(lines):
                raise ValueError(f"Missing sequence for {acc}")
            sequence = lines[i + 1].strip()
            
            # Next line should be annotation
            if i + 2 >= len(lines):
                raise ValueError(f"Missing annotation for {acc}")
            annotation = lines[i + 2].strip()
            
            # Parse record
            record = parse_caid_record(acc, sequence, annotation)
            records.append(record)
            
            i += 3
        else:
            i += 1
    
    return records


def save_caid_fasta(records: list, output_fasta: str):
    """
    Save CAID records to FASTA file.
    """
    output_path = Path(output_fasta)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for record in records:
            f.write(f">{record['id']}\n")
            f.write(f"{record['sequence']}\n")


def save_caid_annotations(records: list, output_csv: str, labels: list = ['structured', 'disordered']):
    """
    Save CAID annotations to CSV file with columns: acc, start, end, label.
    Only regions with known labels (0 or 1) are included.
    """
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for record in records:
        for region in record['annotations']:
            rows.append({
                'acc': record['id'],
                'start': region['start'],
                'end': region['end'],
                'label': labels[1] if region['label'] == '1' else labels[0]
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def process_caid_file(caid_file: str, output_fasta: str, output_csv: str):
    """
    Complete pipeline: read CAID file and generate FASTA + CSV.
    
    Args:
        caid_file: Path to input CAID file
        output_fasta: Path to output FASTA file
        output_csv: Path to output CSV file
    
    Returns:
        Tuple of (number of records, number of annotations)
    """
    print(f"Reading CAID file: {caid_file}")
    records = read_caid_file(caid_file)
    print(f"Parsed {len(records)} records")
    
    print(f"\nSaving FASTA to {output_fasta}")
    save_caid_fasta(records, output_fasta)
    
    print(f"Saving annotations to {output_csv}")
    save_caid_annotations(records, output_csv)
    
    # Count annotations
    total_annotations = sum(len(record['annotations']) for record in records)
    
    return len(records), total_annotations


def read_caid_predictions(caid_file: str) -> dict:
    """
    Read predictions from a CAID file. The files have the columns position, aa, 
    score, [label].
    Returns a dictionary: {acc: {position: {aa: str, score: float, label: str or None}}}
    """
    predictions = {}
    
    with open(caid_file, 'r') as f:
        lines = f.readlines()
    
    current_acc = None
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
        
        if line.startswith('>'):
            current_acc = line[1:]
            predictions[current_acc] = {}
        else:
            parts = line.split()
            if len(parts) >= 3:
                position = int(parts[0])
                aa = parts[1]
                score = float(parts[2])
                label = parts[3] if len(parts) >= 4 else None
                
                predictions[current_acc][position] = {
                    'aa': aa,
                    'score': score,
                    'label': label  # '0' or '1' if present, None otherwise
                }
    
    return predictions


def load_all_predictions(predictions_dir: str) -> dict:
    """
    Load all .caid prediction files from a directory.
    Returns: {method_name: {acc: {position: {aa, score, label}}}}
    """
    predictions_path = Path(predictions_dir)
    all_predictions = {}
    
    for caid_file in sorted(predictions_path.glob('*.caid')):
        method_name = caid_file.stem  # Remove .caid extension
        print(f"Loading predictions from {method_name}...")
        all_predictions[method_name] = read_caid_predictions(str(caid_file))
    
    return all_predictions


def read_fasta_with_labels(fasta_file: str) -> dict:
    """
    Read FASTA file with annotations (format: >acc, sequence, annotation).
    Returns: {acc: {sequence: str, annotation: str}}
    """
    sequences = {}
    
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('>'):
            acc = line[1:]
            
            if i + 1 >= len(lines):
                raise ValueError(f"Missing sequence for {acc}")
            sequence = lines[i + 1].strip()
            
            if i + 2 >= len(lines):
                raise ValueError(f"Missing annotation for {acc}")
            annotation = lines[i + 2].strip()
            
            sequences[acc] = {
                'sequence': sequence,
                'annotation': annotation
            }
            
            i += 3
        else:
            i += 1
    
    return sequences


def create_consolidated_csv(fasta_file: str, predictions_dir: str, output_csv: str):
    """
    Create a consolidated CSV with columns:
    acc, position, aa, label, method1_score, method1_label, method2_score, method2_label, ...
    
    Handles methods with:
    - Only scores (generates label using 0.5 threshold)
    - Scores + labels (uses provided labels)
    
    Args:
        fasta_file: Path to FASTA file with sequences and labels
        predictions_dir: Path to directory with .caid prediction files
        output_csv: Path to output CSV file
    """
    print("=" * 60)
    print("Creating consolidated CSV")
    print("=" * 60)
    
    # Read FASTA with labels
    print(f"\nReading FASTA file: {fasta_file}")
    sequences = read_fasta_with_labels(fasta_file)
    print(f"Found {len(sequences)} sequences")
    
    # Load all predictions
    print(f"\nLoading predictions from: {predictions_dir}")
    all_predictions = load_all_predictions(predictions_dir)
    print(f"Loaded {len(all_predictions)} prediction methods")
    
    # Build rows
    print("\nBuilding consolidated data...")
    rows = []
    
    for acc, seq_data in sequences.items():
        sequence = seq_data['sequence']
        annotation = seq_data['annotation']
        
        for position, aa in enumerate(sequence, start=1):
            row = {
                'acc': acc,
                'position': position,
                'aa': aa,
                'label': annotation[position - 1]  # '0', '1', or '-'
            }
            
            # Add predictions from all methods
            for method_name, predictions in all_predictions.items():
                if acc in predictions and position in predictions[acc]:
                    pred = predictions[acc][position]
                    row[f'{method_name}_score'] = pred['score']
                    
                    # Handle label column
                    if pred['label'] is not None:
                        # Method already has a label, use it
                        row[f'{method_name}_label'] = pred['label']
                    else:
                        # No label in method, generate from score (0.5 threshold)
                        row[f'{method_name}_label'] = '1' if pred['score'] >= 0.5 else '0'
                else:
                    row[f'{method_name}_score'] = None
                    row[f'{method_name}_label'] = None
            
            rows.append(row)
    
    # Create DataFrame and save
    print(f"\nTotal rows: {len(rows)}")
    df = pd.DataFrame(rows)
    
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_csv}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns[:10])}...")  # Show first 10 columns
    
    return df