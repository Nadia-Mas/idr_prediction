# IDR Prediction with ESM2 + MLP

Predict intrinsically disordered regions (IDRs) in protein sequences using ESM2 embeddings and a simple MLP classifier.

## Overview

This project implements a deep learning pipeline for IDR prediction:

1. **Input**: Protein amino acid sequence
2. **Feature extraction**: ESM2 protein language model embeddings (per-residue)
3. **Model**: Multi-layer perceptron (MLP) classifier
4. **Output**: Per-residue disorder probability
5. **Evaluation**: DisProt dataset

## Why This Project?

- **Easy**: Simple architecture, minimal preprocessing
- **Modern**: Uses state-of-the-art protein language models (ESM2)
- **Publishable**: Can be extended and written up as a technical report
- **Resume-worthy**: Demonstrates experience with ML, protein bioinformatics, and deep learning

## Requirements
torch>=2.0.0
transformers>=4.30.0
esm>=0.4.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pyyaml>=6.0
biopython>=1.81
requests>=2.31.0

## Installation

```bash
git clone https://github.com/nadia-mas/idr_prediction.git
cd idr_prediction
pip install -r requirements.txt
```



cd idr_prediction
pip install -r requirements.txt
