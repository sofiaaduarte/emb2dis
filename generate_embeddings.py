"""
python3 generate_embeddings.py --fasta data/raw/CAID3_v3.fasta --output-dir embeddings/ --model ProtT5
"""

import argparse
import os
from src.plms import generate_embeddings_from_fasta

def main():
    parser = argparse.ArgumentParser(description="Generate protein embeddings from FASTA file")
    
    parser.add_argument('--fasta', '-f', type=str, required=True, 
                        help='Path to the input FASTA file')
    
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                        help='Directory to save the generated embeddings (.npy)')
    
    parser.add_argument('--model', '-m', type=str, default='ProtT5',
                        choices=['ESM2', 'ProtT5', 'ProstT5'],
                        help='Protein Language Model (pLM) to use')

    parser.add_argument('--device', '-d', type=str, default='cuda',
                        help='Device to run on (e.g., "cpu", "cuda")')

    args = parser.parse_args()

    # Append model name to output directory
    final_output_dir = os.path.join(args.output_dir, args.model)

    # Create output directory if it doesn't exist
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
        print(f"Created output directory: {final_output_dir}")
 
    print(f"Processing {args.fasta} with {args.model}...")
    
    output_embeddings = generate_embeddings_from_fasta(
        fasta_path=args.fasta,
        output_dir=final_output_dir,
        plm=args.model,
        verbose=True,
        device=args.device
    )
    
    print("Done!")

if __name__ == "__main__":
    main()
