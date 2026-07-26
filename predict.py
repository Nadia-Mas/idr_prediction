"""
Prediction script for new protein sequences.
"""

import torch
import esm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union
import argparse

from models.idr_predictor import IDRPredictor

def load_model(model_path: str, device: torch.device):
    """Load trained IDR prediction model."""
    model = IDRPredictor(
        embedding_dim=480,
        hidden_dims=[256, 128],
        dropout=0.3
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_esm2_embeddings(sequences: List[str], device: torch.device):
    """Get ESM2 embeddings for sequences."""
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model = model.to(device)
    model.eval()
    
    embeddings = []
    for seq in sequences:
        tokens = alphabet.encode(seq)
        tokens = torch.tensor(tokens).unsqueeze(0).to(device)
        
        with torch.no_grad():
            results = model(tokens, repr_layers=[-1])
            emb = results["representations"][-1].squeeze(0)
        embeddings.append(emb)
    
    return embeddings

def predict_disorder(
    sequences: Union[str, List[str]],
    model_path: str = "checkpoints/idr_model.pt",
    device: torch.device = None
) -> List[np.ndarray]:
    """
    Predict disorder probabilities for protein sequences.
    
    Args:
        sequences: Single sequence string or list of sequences
        model_path: Path to trained model
        device: torch device
        
    Returns:
        List of arrays with per-residue disorder probabilities
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if isinstance(sequences, str):
        sequences = [sequences]
    
    # Load model
    model = load_model(model_path, device)
    
    # Get embeddings
    embeddings = get_esm2_embeddings(sequences, device)
    
    # Predict
    predictions = []
    for emb in embeddings:
        emb = emb.unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            logits = model(emb)
            probs = torch.softmax(logits, dim=-1)
            disorder_prob = probs[0, :, 1].cpu().numpy()  # Probability of class 1 (disorder)
        predictions.append(disorder_prob)
    
    return predictions

def plot_disorder_profile(sequence: str, disorder_probs: np.ndarray, save_path: str = None):
    """Plot disorder probability along the sequence."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Disorder probability plot
    axes[0].plot(disorder_probs, color='blue', linewidth=2)
    axes[0].axhline(y=0.5, color='red', linestyle='--', label='Threshold (0.5)')
    axes[0].fill_between(
        range(len(disorder_probs)),
        0,
        disorder_probs,
        where=(disorder_probs > 0.5),
        color='red',
        alpha=0.3,
        label='Predicted IDR'
    )
    axes[0].set_xlabel('Residue Position')
    axes[0].set_ylabel('Disorder Probability')
    axes[0].set_title('IDR Prediction Profile')
    axes[0].legend()
    axes[0].set_ylim([0, 1])
    
    # Amino acid sequence display (abbreviated)
    seq_display = ' '.join(list(sequence))
    axes[1].text(0.5, 0.5, seq_display, 
                fontsize=8, fontfamily='monospace',
                horizontalalignment='center',
                verticalalignment='center',
                transform=axes[1].transAxes)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    axes[1].axis('off')
    axes[1].set_title('Amino Acid Sequence')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Predict IDRs from protein sequences')
    parser.add_argument('--sequence', type=str, help='Protein sequence')
    parser.add_argument('--fasta', type=str, help='FASTA file with sequences')
    parser.add_argument('--model', type=str, default='checkpoints/idr_model.pt', 
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV file')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    # Get sequences
    sequences = []
    names = []
    
    if args.sequence:
        sequences.append(args.sequence)
        names.append('query')
    elif args.fasta:
        from Bio import SeqIO
        for record in SeqIO.parse(args.fasta, 'fasta'):
            sequences.append(str(record.seq))
            names.append(record.id)
    else:
        # Example sequences
        sequences = [
            "MDDNHYPHHHHNHHNHHSTSGGCGESQFTTKLSVNTFARTHPMIQNDLIDLDLISGSAFTMKSKSQQ",
            "PADRDLSSPFGSTVPGVGPNAAAASNAAAAAAAAATAGSNKHQTPPTTFR",
        ]
        names = ['example1', 'example2']
    
    # Predict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions = predict_disorder(sequences, args.model, device)
    
    # Save results
    results = []
    for name, seq, probs in zip(names, sequences, predictions):
        results.append({
            'name': name,
            'sequence': seq,
            'length': len(seq),
            'disorder_probs': probs.tolist(),
            'disorder_fraction': float(np.mean(probs > 0.5)),
            'max_disorder': float(np.max(probs))
        })
    
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    
    # Print summary
    print("\nPrediction Summary:")
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  Length: {r['length']}")
        print(f"  Disorder fraction: {r['disorder_fraction']:.2%}")
        print(f"  Max disorder probability: {r['max_disorder']:.3f}")
    
    # Plot
    if args.plot:
        for name, seq, probs in zip(names, sequences, predictions):
            plot_disorder_profile(seq, probs, f"plots/{name}_profile.png")

if __name__ == "__main__":
    main()
