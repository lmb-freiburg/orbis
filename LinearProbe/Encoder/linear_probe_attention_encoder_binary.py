import os
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import wandb

# Fix paths to recognize sister modules if executing from nested directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


class CachedFeatureDataset(Dataset):
    def __init__(self, cache_path):
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Target embedding cache not found at: {cache_path}")
            
        data = torch.load(cache_path, map_location="cpu")
        features = data['features']  # Input shape: [Batch, 32, 18, 32]
        self.labels = data['labels'].long() 
        self.ids = data['ids']
        
        # --- Spatial-Temporal Restructuring ---
        # 1. Permute the [-3] channel dimension to the trailing position
        # [B, 32, 18, 32] -> [B, 18, 32, 32]
        features = features.permute(0, 2, 3, 1)
        
        # 2. Flatten spatial dimensions to generate the sequence length
        # [B, 18, 32, 32] -> [B, 576, 32]
        self.features = features.reshape(features.size(0), 576, 32)
        
        if self.labels.dim() > 1:
            self.labels = self.labels.squeeze()
            
        print(f'Loaded {cache_path} | Restructured Shape - Features: {self.features.shape} , Labels: {self.labels.shape}')
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.ids[idx]


class AttentionProbe(nn.Module):
    def __init__(self, input_dim, num_classes=2, num_heads=8):
        super().__init__()
        # 1. Learnable query token acting as the context summarizer
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # 2. Multi-head Cross Attention Layer
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        
        # 3. Classifier mapping contextual features to class space
        self.classifier = nn.Linear(input_dim, num_classes)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x expected shape: [Batch, 576, 32]
        B = x.size(0)
        
        # Norm 1
        x = self.norm1(x)

        # Expand query token across the batch size
        q = self.query.expand(B, -1, -1)
        
        # Perform cross-attention pooling over the sequential tokens
        attn_out, attn_weights = self.attn(query=q, key=x, value=x)
        
        # Compress sequence dimension: [Batch, 1, 32] -> [Batch, 32]
        pooled_features = attn_out.squeeze(1)

        # Norm 2
        pooled_features = self.norm2(pooled_features)
        
        # Logit classification output
        logits = self.classifier(pooled_features)
        return logits


def train_linear_probe():
    # 1. Initialize W&B run (Config populated dynamically by the Sweep Agent)
    wandb.init()
    config = wandb.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load the restructured datasets from the caching directory
    cache_dir = getattr(config, "cache_dir", "./cached_features")
    train_dataset = CachedFeatureDataset(os.path.join(cache_dir, "train_unpooled_embeddings.pt"))
    val_dataset = CachedFeatureDataset(os.path.join(cache_dir, "val_unpooled_embeddings.pt"))

    print(f'------- Train: {len(train_dataset)} | Val: {len(val_dataset)} ---------')
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 3. Initialize Attention Probe with embed_dim=32
    model = AttentionProbe(input_dim=32, num_heads=8).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    )
    
    # Early Stopping Setup
    best_val_loss = float('inf')
    patience = config.early_stopping_patience
    patience_counter = 0
    epochs = 50 
    
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct = 0
        total = 0
        
        for idx, (features, labels, _) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # 5. Validation Loop
        model.eval()
        total_val_loss = 0
        
        all_val_labels = []
        all_val_preds = []
        all_val_probs = []

        with torch.no_grad():
            for features, labels, _ in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()
                
                _, predicted = outputs.max(1)
                probs = F.softmax(outputs, dim=1)[:, 1]
                
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())
                
        # 6. Metrics Calculations
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(all_val_labels, all_val_preds) * 100
        val_precision = precision_score(all_val_labels, all_val_preds, zero_division=0) * 100
        val_recall = recall_score(all_val_labels, all_val_preds, zero_division=0) * 100
        
        try:
            val_auc = roc_auc_score(all_val_labels, all_val_probs)
        except ValueError:
            val_auc = float('nan')
            
        cm = confusion_matrix(all_val_labels, all_val_preds)
        
        # 7. Reporting & Logging
        print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"--> Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | AUC: {val_auc:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_acc,
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_auc": val_auc
        })

        # 8. Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Early Stopping Counter: {patience_counter} / {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered. Halting training.")
                break


if __name__ == "__main__":
    # Define the Hyperparameter Sweep Configuration matching encoder parameters
    sweep_config = {
        'method': 'bayes', # Bayesian optimization 
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 1e-3
            },
            'weight_decay': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.1
            },
            'batch_size': {
                'values': [16, 32, 64]
            },
            'beta1': {
                'values': [0.9, 0.95]
            },
            'beta2': {
                'values': [0.99, 0.999]
            },
            'early_stopping_patience': {
                'value': 5
            },
            'cache_dir': {
                'value': './cached_features'
            }
        }
    }
    # # Best Hyper Param-Setting - misunderstood-sweep-20
    #     batch_size: 16
    #     beta1: 0.95
    #     beta2:0.999
    #     early_stopping_patience:5
    #     learning_rate:0.00007114681159456426
    #     weight_decay:0.012770392467248684
    # # Best Summary metrics for above Hyperparameters setting
    # {
    #     "_step": 48,
    #     "epoch": 49,
    #     "_wandb.runtime": 23,
    #     "val_auc": 0.6660081491542166,
    #     "_runtime": 23,
    #     "val_loss": 0.6123684744040171,
    #     "_timestamp": 1784542445.670143,
    #     "train_loss": 0.6110577013757493,
    #     "val_recall": 53.84615384615385,
    #     "val_accuracy": 66.66666666666666,
    #     "val_precision": 73.13432835820896,
    #     "train_accuracy": 65.13888888888889
    # }



    # Initialize W&B Sweep
    sweep_id = wandb.sweep(sweep_config, project="orbis-encoder-attention-probe")

    # Launch sweep agent
    wandb.agent(sweep_id, function=train_linear_probe, count=20)