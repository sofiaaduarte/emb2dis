"""
Predict disorder from protein embeddings using a trained model
"""

import argparse
import numpy as np
import torch as tr
import pandas as pd
from pathlib import Path
from src.model import BaseModel
from src.utils import (
    ConfigLoader,
    predict_sliding_window,
    get_embedding_size,
    calculate_disorder_percentage,
)
from src.plot import plot_disorder_prediction

def handle():
    """
    Function wrapper that loads command args and calls the main function
    """
    protein_id = "DP04142"
    # Options are: 
    # "DP04142"
    # "DP04179", 
    # "DP04199",
    # "DP04219"
    return main(protein_id)

def main(protein_id):
    """
    Main prediction function
    """ 
    device = 'cpu'
    verbose = True
    language_model = 'ESM2'
    output_dir = "results_prueba/"

    # Set up model path, config and weights ------------------------------------
    model_dir = Path(f"model/{language_model}/model0/")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    config_path = model_dir / "config.yaml"
    weights_path = model_dir / "weights.pk"

    # Load model configuration -------------------------------------------------
    if verbose:
        print(f"Model directory: {model_dir}")
        print(f"Loading configuration from: {config_path}")
    config_loader = ConfigLoader(model_path=str(config_path))
    config = config_loader.load()
    window_len = config.get("win_len", 13)
    threshold = config.get("threshold", 0.5)

    # Initialize model ---------------------------------------------------------
    if verbose:
        print(f"Loading model from: {weights_path}")
    categories = ["structured", "disordered"]
    model = BaseModel(
        len(categories),
        lr=config["lr"],
        device=device,
        emb_size=get_embedding_size(config.get("plm", "ESM2")),
        filters=config["filters"],
        kernel_size=config["kernel_size"],
        num_layers=config["n_resnet"],
    )
    model.load_state_dict(tr.load(weights_path, map_location=device))
    model.eval()

    # Load embedding from .npy file in data/ directory -------------------------
    print(f"\nLoading embedding for {protein_id}")
    data_dir = Path("data")
    emb_file = data_dir / f"{protein_id}.npy"
    
    if not emb_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {emb_file}")
    
    # Load embedding tensor
    emb_array = np.load(emb_file)
    emb = tr.tensor(emb_array, dtype=tr.float32)
    
    if verbose:
        print(f"Loaded {protein_id}: shape {emb_array.shape}")

    # Predict disorder and save results ------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n--- Processing Protein: {protein_id} ---")
        print(f"Sequence length: {emb.shape[1]} residues")

    # Predict ------------------------------------------------------------------
    if verbose:
        print(f"Predicting disorder (window={window_len})")
    centers, predictions = predict_sliding_window(
        model,
        emb,
        window_len,
        step=1,
        use_softmax=config.get("soft_max", True),
        median_filter_size=None,  # No smoothing
    )

    # Calculate disorder percentage
    stats = calculate_disorder_percentage(predictions, threshold=threshold)

    # Print results
    print(f"\nDISORDER PREDICTION RESULTS FOR: {protein_id}")
    print(f"Total residues:        {stats['total_residues']}")
    print(f"Disordered residues:   {stats['disordered_residues']}")
    print(f"Disorder percentage:   {stats['disorder_percentage']:.2f}%")

    # Save outputs -------------------------------------------------------------

    # Save plot
    output_plot = output_dir / f"{protein_id}_{language_model}_plot.png"
    plot_disorder_prediction(
        centers,
        predictions,
        protein_id,
        threshold=threshold,
        output_path=output_plot,
    )

    # Save predictions to CSV
    output_csv = output_dir / f"{protein_id}_{language_model}_predictions.csv"
    df = pd.DataFrame(
        {
            "position": centers,
            "disordered_score": predictions[:, 1].numpy(),
            "predicted_label": (predictions[:, 1] > threshold).numpy().astype(int),
        }
    )
    df.to_csv(output_csv, index=False)

    if verbose:
        print(f"Plot saved to: {output_plot}")
        print(f"Predictions saved to: {output_csv}")

    stats = {
        'total_residues': stats['total_residues'],
        'disordered_residues': stats['disordered_residues'],
        'disorder_percentage': stats['disorder_percentage'],
        'protein_id': protein_id
    }

    return stats, output_csv, output_plot


if __name__ == "__main__":
    handle()

