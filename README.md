# IDR Prediction with ESM-2 and Deep Learning

A reproducible deep learning pipeline for predicting **Intrinsically Disordered Regions (IDRs)** in protein sequences using **ESM-2 protein language model embeddings** and a **Multi-Layer Perceptron (MLP)** classifier.

---

## Project Overview

Intrinsically Disordered Regions (IDRs) are protein regions that do not adopt a stable three-dimensional structure under physiological conditions but play essential roles in signaling, regulation, molecular recognition, and disease.

This project leverages large protein language models to generate contextual residue-level embeddings and trains a lightweight neural network to classify each residue as **ordered** or **disordered**.

The complete pipeline is fully reproducible and includes:

- Automatic DisProt dataset download and preprocessing
- Protein-level train/validation/test splitting
- Residue-level ESM-2 embeddings
- Embedding caching for efficient training
- MLP classifier
- Early stopping and checkpointing
- Comprehensive evaluation and visualization

---

## Pipeline

```
Protein Sequence
        │
        ▼
 DisProt Dataset
        │
        ▼
 Data Preprocessing
        │
        ▼
 Protein-Level Split
        │
        ▼
 ESM-2 Embedding Extraction
        │
        ▼
 Cached Residue Embeddings
        │
        ▼
 Multi-Layer Perceptron
        │
        ▼
 Residue Disorder Prediction
        │
        ▼
 Performance Evaluation
```

---

## Features

- Automatic DisProt dataset processing
- Hugging Face ESM-2 protein language model
- Frozen residue embeddings
- Protein-level data split (prevents data leakage)
- Embedding caching to accelerate experiments
- Early stopping
- Model checkpointing
- Publication-ready plots
- Easily extendable to other protein prediction tasks

---

## Repository Structure

```
idr_prediction/
│
├── IDR_full_pipeline.ipynb          # Complete training notebook
│
├── data/
│   ├── raw/
│   └── processed/
│
├── embeddings/                      # Generated automatically
├── checkpoints/                     # Saved models
├── plots/                           # Training figures
├── results/                         # Evaluation metrics
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Model

### Feature Extractor

- **Model:** Facebook ESM-2
- **Version:** esm2_t12_35M_UR50D
- Residue-level contextual embeddings
- Frozen during classifier training

### Classifier

A lightweight Multi-Layer Perceptron (MLP)

- Fully connected layers
- Dropout regularization
- Binary residue classification
- Weighted Binary Cross-Entropy loss
- Adam optimizer
- Early stopping

---

## Dataset

**DisProt**

DisProt is a manually curated database containing experimentally validated intrinsically disordered proteins and regions.

Each residue is labeled as:

- 0 → Ordered
- 1 → Disordered

The dataset is automatically downloaded and processed within the notebook.

---

## Evaluation Metrics

The model is evaluated using:

- Area Under ROC Curve (AUROC)
- Area Under Precision-Recall Curve (AUPRC)
- F1 Score
- Precision
- Recall
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix

---

## Requirements

```text
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
biopython>=1.81
requests>=2.31.0
PyYAML>=6.0
```

> **Note:** This project uses the Hugging Face implementation of ESM-2. The `esm` package is **not required**.

---

## Installation

Clone the repository

```bash
git clone https://github.com/nadia-mas/idr_prediction.git
cd idr_prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

Open the notebook

```
IDR_full_pipeline.ipynb
```

Run all cells sequentially.

The notebook will automatically:

1. Download and preprocess DisProt
2. Create train/validation/test splits
3. Extract ESM-2 embeddings
4. Cache embeddings
5. Train the MLP
6. Save checkpoints
7. Evaluate the model
8. Generate figures

---

## Output

Training generates

```
checkpoints/
```

- Best model checkpoint

```
plots/
```

- Training loss
- Validation curves
- Confusion matrix

```
results/
```

- Prediction outputs
- Evaluation metrics

---

## Future Improvements

Possible future extensions include:

- Fine-tuning ESM-2
- Transformer-based residue classifiers
- Conditional Random Fields (CRFs)
- Ensemble learning
- Multi-task prediction of protein properties
- Cross-dataset benchmarking
- Model interpretability (Integrated Gradients, SHAP)

---

## Citation

If you use this repository in your research, please cite:

```
Masoumi, F. S.
IDR Prediction with ESM-2 and Deep Learning.
GitHub Repository.
```

---

## Acknowledgements

- Meta AI for the ESM-2 protein language model
- Hugging Face Transformers
- DisProt Consortium
- PyTorch
