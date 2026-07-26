"""
Training script for IDR prediction model.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data.download_disprot import DisProtDownloader
from data.dataset import create_dataloaders
from models.idr_predictor import create_model, count_parameters

def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        embeddings = batch['embeddings'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(embeddings)
        
        # Reshape for loss (ignore padding with -100)
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1)
        
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions for metrics
        preds = logits.argmax(dim=-1)
        mask = labels != -100
        all_preds.extend(preds[mask].cpu().numpy())
        all_labels.extend(labels[mask].cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            embeddings = batch['embeddings'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(embeddings)
            
            # Reshape
            logits = logits.reshape(-1, logits.shape[-1])
            labels = labels.reshape(-1)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            mask = labels != -100
            all_preds.extend(preds[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())
            all_probs.extend(probs[mask, 1].cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # ROC-AUC for positive class
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except:
        roc_auc = 0.0
    
    return avg_loss, accuracy, f1, roc_auc

def plot_training_history(history, save_path: str = "plots/training_history.png"):
    """Plot training and validation metrics."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Loss')
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].set_title('Accuracy')
    
    # F1
    axes[1, 0].plot(history['train_f1'], label='Train F1')
    axes[1, 0].plot(history['val_f1'], label='Val F1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].set_title('F1 Score')
    
    # ROC-AUC (validation only)
    axes[1, 1].plot(history['val_roc_auc'], label='Val ROC-AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ROC-AUC')
    axes[1, 1].legend()
    axes[1, 1].set_title('ROC-AUC')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training loop."""
    # Load config
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Download and prepare data
    print("\n=== Downloading DisProt Data ===")
    downloader = DisProtDownloader()
    entries = downloader.get_entries(limit=2000)
    
    if not entries:
        print("Using cached dataset if available...")
        # Try loading from cache
        try:
            df = pd.read_pickle("data/raw/disprot_dataset.pkl")
        except:
            print("No cached dataset found. Please check your internet connection.")
            return
    else:
        df = downloader.create_dataset(entries)
        downloader.save_dataset(df)
    
    # Create dataloaders
    print("\n=== Creating Dataloaders ===")
    train_loader, val_loader, test_loader = create_dataloaders(
        df,
        batch_size=config['training']['batch_size'],
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        max_length=config['data']['max_sequence_length'],
        use_esm2=True
    )
    
    # Create model
    print("\n=== Creating Model ===")
    model = create_model(
        model_type="mlp",
        embedding_dim=config['model']['embedding_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout']
    )
    model = model.to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print("\n=== Training ===")
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'val_roc_auc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Evaluate
        val_loss, val_acc, val_f1, val_roc_auc = evaluate(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['val_roc_auc'].append(val_roc_auc)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, ROC-AUC: {val_roc_auc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), config['output']['model_save_path'])
            print(f"Saved best model to {config['output']['model_save_path']}")
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Final evaluation on test set
    print("\n=== Final Evaluation on Test Set ===")
    model.load_state_dict(torch.load(config['output']['model_save_path']))
    test_loss, test_acc, test_f1, test_roc_auc = evaluate(
        model, test_loader, criterion, device
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    
    # Plot training history
    plot_training_history(history, config['output']['plots_dir'] + "training_history.png")
    
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()
