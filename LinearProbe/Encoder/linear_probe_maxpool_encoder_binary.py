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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


class CachedFeatureDataset(Dataset):
    def __init__(self, cache_path):
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Target embedding cache not found at: {cache_path}")
            
        data = torch.load(cache_path, map_location="cpu")
        features = data['features']           # Raw Cache Shape: [Batch, 32, 18, 32]
        self.labels = data['labels'].long() 
        
        # --- Spatial-Temporal Restructuring ---
        # 1. Permute the [-3] channel dimension to the trailing position: [B, 32, 18, 32] -> [B, 18, 32, 32]
        features = features.permute(0, 2, 3, 1)
        
        # 2. Flatten intermediate spatial dimensions: [B, 18, 32, 32] -> [B, 576, 32]
        self.features = features.reshape(features.size(0), 576, 32)
        
        if self.labels.dim() > 1:
            self.labels = self.labels.squeeze()
            
        print(f'Loaded {cache_path} | Restructured Shape - Features: {self.features.shape} | Labels: {self.labels.shape}')
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.norm(x))


def train_linear_probe():
    # 1. Initialize W&B run (Config is populated by the Sweep agent)
    wandb.init()
    config = wandb.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load Cached Data (Pointing to unpooled combined embeddings)
    train_dataset = CachedFeatureDataset("./cached_features/train_unpooled_embeddings.pt")
    val_dataset = CachedFeatureDataset("./cached_features/val_unpooled_embeddings.pt")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 3. Initialize Probe with target Embedding Dimension
    hidden_dim = 32 

    model = LinearProbe(input_dim=hidden_dim).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Use HPO configs for AdamW
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
        
        for idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            # Global Max Pooling along the spatial sequence dimension of 576 (dim=1)
            # torch.max returns a namedtuple (values, indices). We grab the values [0].
            pooled_features = torch.max(features, dim=1)[0]  # Shape: [Batch, 32]

            optimizer.zero_grad()
            outputs = model(pooled_features)
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
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)

                # Global Max Pooling along the sequence dimension of 576 (dim=1)
                pooled_features = torch.max(features, dim=1)[0]  # Shape: [Batch, 32]

                outputs = model(pooled_features)
                
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()
                
                _, predicted = outputs.max(1)
                probs = F.softmax(outputs, dim=1)[:, 1]
                
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())
                
        # 6. Calculate Metrics
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(all_val_labels, all_val_preds) * 100
        val_precision = precision_score(all_val_labels, all_val_preds, zero_division=0) * 100
        val_recall = recall_score(all_val_labels, all_val_preds, zero_division=0) * 100
        
        try:
            val_auc = roc_auc_score(all_val_labels, all_val_probs)
        except ValueError:
            val_auc = float('nan')
            
        # 7. Print and Log to W&B
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

        # 8. Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            checkpoint_dir = "./checkpoints/encoder"
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_encoder_maxpool_probe_binary.pt"))
            print(f">>> Saved new best encoder maxpool probe model to '{checkpoint_dir}'! <<<")
        else:
            patience_counter += 1
            print(f"Early Stopping Counter: {patience_counter} / {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered. Halting training.")
                break 

if __name__ == "__main__":
    # Define the Hyperparameter Sweep Configuration
    sweep_config = {
        'method': 'bayes', 
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
            }
        }
    }

    # # Best Hyper Param-Setting - upbeat-sweep-20
    #     batch_size:32
    #       beta1: 0.95
    # beta2: 0.999
    # early_stopping_patience:5
    # learning_rate: 0.000915962842484255
    # weight_decay: 0.04801436623071827


    # # Best Summary metrics for above Hyperparameters setting
    # {
    #   "_step": 22,
    #   "epoch": 23,
    #   "_wandb.runtime": 3,
    #   "val_auc": 0.6057537967650327,
    #   "_runtime": 3,
    #   "val_loss": 0.6746813456217448,
    #   "_timestamp": 1784199880.041634,
    #   "train_loss": 0.6837285109188246,
    #   "val_recall": 59.34065934065934,
    #   "val_accuracy": 60.55555555555555,
    #   "val_precision": 61.36363636363637,
    #   "train_accuracy": 55
    # }
    
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="orbis-encoder-linear-maxpool-probe")

    # Run the sweep agent (20 iterations)
    wandb.agent(sweep_id, function=train_linear_probe, count=20)