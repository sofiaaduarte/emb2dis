"""
Predict disorder from protein embeddings using a trained model
"""
import yaml
import numpy as np
import torch as tr
from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm
from src.model import BaseModel
from src.utils import ConfigLoader, predict_sliding_window

CONFIG_PATH = "config/predict.yaml"

def main():

    # Load predict config
    with open(CONFIG_PATH) as f:
        args = yaml.safe_load(f)

    # Set up model path, config and weights ------------------------------------
    model_dir = Path(args['model_dir'])
    print(f"Model directory: {model_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    config_path = model_dir / 'config.yaml'
    weights_path = model_dir / 'weights.pk'
    
    # Load model configuration -------------------------------------------------
    # print(f"Loading configuration from: {config_path}")
    config_loader = ConfigLoader(model_path=str(config_path))
    config = config_loader.load()

    device = config.get('device', 'cpu')
    threshold = config.get('threshold', 0.5)
    
    # Initialize model ---------------------------------------------------------
    # print(f"Loading model from: {weights_path}")
    categories = config.get('categories', ['structured', 'disordered'])

    model = BaseModel(
        len(categories),
        emb_size=config['emb_size'][config['pLM']],
        lr=config['lr'],
        device=device,
        p_dropout=config['p_dropout'],
        filters=config['filters'],
        kernel_size=config['kernel_size'],
        num_layers=config['n_resnet']
    )
    model.load_state_dict(tr.load(weights_path, map_location=device))
    model.eval()
    
    # Load embeddings from folder using IDs from FASTA -----------------------
    emb_dir = Path(config['emb_path']) / config['pLM']
    if not emb_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {emb_dir}")

    with open(args['fasta']) as f:
        records = list(SeqIO.parse(f, "fasta"))

    results = []
    for record in records:
        protein_id = record.id
        emb_file = emb_dir / f"{protein_id}.npy"
        if not emb_file.exists():
            print(f"Warning: embedding not found for {protein_id}, skipping.")
            continue
        emb = np.load(emb_file)
        emb = tr.tensor(emb, dtype=tr.float32)
        results.append((emb, protein_id, str(record.seq)))

    print(f"\nLoaded {len(results)}/{len(records)} embeddings from: {emb_dir}")
    
    # Predict disorder for all the proteins and save results -------------------

    all_rows = []

    # For each protein embedding and ID
    for emb, protein_id, sequence in tqdm(results, desc="Predicting"):

        centers, predictions = predict_sliding_window(
            model,
            emb,
            config['win_len'],
            step=1, 
            use_softmax=config['soft_max'],
            median_filter_size=None  # No smoothing
        )

        scores = predictions[:, 1].numpy()
        labels = (scores > threshold).astype(int)
        all_rows.append((protein_id, centers, sequence, scores, labels))

    # Save all predictions to a single file -----------------------------------
    output_csv = model_dir / "emb2dis.caid"
    with open(output_csv, 'w') as out:
        for protein_id, centers, sequence, scores, labels in all_rows:
            out.write(f">{protein_id}\n")
            for idx, score, label in zip(centers, scores, labels):
                aa = sequence[idx]          # centers are 0-based
                pos = idx + 1               # 1-based position for output
                out.write(f"{pos}\t{aa}\t{score:.3f}\t{label}\n")
    print(f"\nPredictions saved to: {output_csv}")

if __name__ == '__main__':
    main()