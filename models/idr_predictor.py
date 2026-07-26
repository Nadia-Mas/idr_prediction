"""
MLP model for IDR prediction from ESM2 embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class IDRPredictor(nn.Module):
    """
    MLP classifier for per-residue IDR prediction.
    
    Takes ESM2 embeddings as input and predicts disorder probability
    for each residue position.
    """
    
    def __init__(
        self,
        embedding_dim: int = 480,  # ESM2-35M dimension
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super(IDRPredictor, self).__init__()
        
        layers = []
        prev_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            embeddings: (batch_size, seq_len, embedding_dim)
            
        Returns:
            logits: (batch_size, seq_len, num_classes)
        """
        # Apply MLP to each position independently
        batch_size, seq_len, emb_dim = embeddings.shape
        embeddings = embeddings.reshape(-1, emb_dim)
        logits = self.mlp(embeddings)
        logits = logits.reshape(batch_size, seq_len, -1)
        
        return logits

class IDRPredictorWithAttention(nn.Module):
    """
    IDR predictor with self-attention for context.
    
    Adds a transformer encoder layer to capture contextual information
    across the sequence before the MLP.
    """
    
    def __init__(
        self,
        embedding_dim: int = 480,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
        num_heads: int = 8,
        num_layers: int = 2,
        num_classes: int = 2
    ):
        super(IDRPredictorWithAttention, self).__init__()
        
        # Transformer encoder for context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # MLP head
        layers = []
        prev_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            embeddings: (batch_size, seq_len, embedding_dim)
            
        Returns:
            logits: (batch_size, seq_len, num_classes)
        """
        # Apply transformer for context
        contextualized = self.transformer(embeddings)
        
        # Apply MLP
        batch_size, seq_len, emb_dim = contextualized.shape
        contextualized = contextualized.reshape(-1, emb_dim)
        logits = self.mlp(contextualized)
        logits = logits.reshape(batch_size, seq_len, -1)
        
        return logits

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_model(
    model_type: str = "mlp",
    embedding_dim: int = 480,
    **kwargs
) -> nn.Module:
    """Factory function for creating IDR prediction models."""
    if model_type == "mlp":
        return IDRPredictor(embedding_dim=embedding_dim, **kwargs)
    elif model_type == "attention":
        return IDRPredictorWithAttention(embedding_dim=embedding_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
