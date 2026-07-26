"""
Dataset classes for IDR prediction.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from transformers import AutoTokenizer
import esm

class IDRDataset(Dataset):
    """Dataset for IDR prediction with ESM2 embeddings."""
    
    def __init__(
        self,
        sequences: List[str],
        labels: List[List[int]],
        tokenizer,
        max_length: int = 1024
    ):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        labels = self.labels[idx]
        
        # Tokenize sequence
        encoding = self.tokenizer(
            seq,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        
        # Get input_ids and attention_mask
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Pad labels to max_length (pad with -100 for ignore)
        padded_labels = torch.full((self.max_length,), -100, dtype=torch.long)
        seq_len = min(len(labels), self.max_length)
        padded_labels[:seq_len] = torch.tensor(labels[:seq_len], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': padded_labels,
            'sequence_length': len(seq)
        }

class ESM2EmbeddingDataset(Dataset):
    """Dataset that precomputes ESM2 embeddings."""
    
    def __init__(
        self,
        sequences: List[str],
        labels: List[List[int]],
        model,
        alphabet,
        max_length: int = 1024,
        batch_size: int = 8
    ):
        self.sequences = sequences
        self.labels = labels
        self.model = model
        self.alphabet = alphabet
        self.max_length = max_length
        self.batch_size = batch_size
        self.embeddings = None
        
    def compute_embeddings(self):
        """Compute ESM2 embeddings for all sequences."""
        embeddings = []
        
        for i in range(0, len(self.sequences), self.batch_size):
            batch_seqs = self.sequences[i:i + self.batch_size]
            batch_labels = self.labels[i:i + self.batch_size]
            
            # Prepare batch
            batch_tokens = []
            for seq in batch_seqs:
                # Truncate if necessary
                seq = seq[:self.max_length]
                tokens = self.alphabet.encode(seq)
                batch_tokens.append(tokens)
            
            # Pad batch
            max_len = max(len(tokens) for tokens in batch_tokens)
            padded_tokens = torch.full(
                (len(batch_tokens), max_len),
                self.alphabet.padding_idx,
                dtype=torch.long
            )
            for j, tokens in enumerate(batch_tokens):
                padded_tokens[j, :len(tokens)] = torch.tensor(tokens)
            
            # Get embeddings
            with torch.no_grad():
                results = self.model(padded_tokens, repr_layers=[-1])
                batch_embeddings = results["representations"][-1]  # Last layer
            
            embeddings.append(batch_embeddings)
        
        self.embeddings = torch.cat(embeddings, dim=0)
        return self.embeddings
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.embeddings is None:
            raise ValueError("Call compute_embeddings() first")
        
        emb = self.embeddings[idx]
        labels = self.labels[idx]
        
        # Pad labels
        seq_len = emb.shape[0]
        padded_labels = torch.full((self.max_length,), -100, dtype=torch.long)
        padded_labels[:min(len(labels), self.max_length)] = torch.tensor(
            labels[:min(len(labels), self.max_length)], dtype=torch.long
        )
        
        # Pad embeddings
        if seq_len < self.max_length:
            pad_size = self.max_length - seq_len
            emb = torch.cat([
                emb,
                torch.zeros(pad_size, emb.shape[1])
            ], dim=0)
        else:
            emb = emb[:self.max_length]
        
        return {
            'embeddings': emb,
            'labels': padded_labels,
            'sequence_length': seq_len
        }

def create_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    max_length: int = 1024,
    use_esm2: bool = True
):
    """Create train/val/test dataloaders."""
    
    # Split data
    n = len(df)
    indices = np.random.permutation(n)
    
    train_end = int(train_split * n)
    val_end = int((train_split + val_split) * n)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]
    test_df = df.iloc[test_indices]
    
    if use_esm2:
        # Load ESM2 model for embeddings
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        model.eval()
        
        # Create datasets
        train_dataset = ESM2EmbeddingDataset(
            train_df['sequence'].tolist(),
            train_df['labels'].tolist(),
            model, alphabet, max_length
        )
        val_dataset = ESM2EmbeddingDataset(
            val_df['sequence'].tolist(),
            val_df['labels'].tolist(),
            model, alphabet, max_length
        )
        test_dataset = ESM2EmbeddingDataset(
            test_df['sequence'].tolist(),
            test_df['labels'].tolist(),
            model, alphabet, max_length
        )
    else:
        # Use tokenizer for on-the-fly encoding
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
        
        train_dataset = IDRDataset(
            train_df['sequence'].tolist(),
            train_df['labels'].tolist(),
            tokenizer, max_length
        )
        val_dataset = IDRDataset(
            val_df['sequence'].tolist(),
            val_df['labels'].tolist(),
            tokenizer, max_length
        )
        test_dataset = IDRDataset(
            test_df['sequence'].tolist(),
            test_df['labels'].tolist(),
            tokenizer, max_length
        )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader
