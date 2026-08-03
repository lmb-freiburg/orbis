import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split # Added random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import wandb # Added Weights & Biases

class CachedFeatureDataset(Dataset):
    def __init__(self, cache_path):
        data = torch.load(cache_path)
        self.features = data['features']
        self.labels = data['labels'].long() 
        # If you saved IDs, you can also load self.ids = data['ids'] here
        print(f'Loaded {cache_path} | Shape - {self.features.shape} , {self.labels.shape}')
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Add self.ids[idx] here if you added IDs to your caching script!
        return self.features[idx], self.labels[idx]

class AttentionProbe(nn.Module):
    def __init__(self, input_dim, num_classes=2, num_heads=8):
        super().__init__()
        # 1. Learnable query token (Think of this as the "Detective")
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # 2. Multi-head Attention Layer
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        
        # 3. Final linear classifier
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x is your unpooled cached features. Shape: [Batch, 576, 768]
        B = x.size(0)
        
        # Expand our single query token to match the batch size
        q = self.query.expand(B, -1, -1)
        
        # Cross-Attention
        attn_out, attn_weights = self.attn(query=q, key=x, value=x)
        
        # Squeeze out the sequence dimension: [Batch, 1, 768] -> [Batch, 768]
        pooled_features = attn_out.squeeze(1)
        
        # Pass the attended features into the final classifier
        logits = self.classifier(pooled_features)
        
        return logits

def train_linear_probe():
    # 1. Initialize W&B run (Config is populated by the Sweep agent)
    wandb.init()
    config = wandb.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load the Single Partial Cached Data
    full_dataset = CachedFeatureDataset("./cached_features/train_block18_unpooled_partial.pt")

    # 3. Perform 80/20 Split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # Use a fixed generator seed so the split is identical across all sweep runs
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f'------- Train Split Size: {train_size} | Val Split Size: {val_size} ---------')
    
    # Drop last to avoid batch size mismatch issues in edge cases
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    # val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 4. Initialize Attention Probe
    hidden_dim = 768 
    model = AttentionProbe(input_dim=hidden_dim, num_heads=8).to(device)
    
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
    epochs = 50 # Max epochs, but early stopping handles halting
    
    # 5. Training Loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct = 0
        total = 0
        
        for idx, (features, labels, *_) in enumerate(train_loader):
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
        
        # 6. Validation Loop
        model.eval()
        total_val_loss = 0
        
        all_val_labels = []
        all_val_preds = []
        all_val_probs = []

        with torch.no_grad():
            for features, labels, *_ in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                
                # Calculate Validation Loss for Early Stopping
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()
                
                _, predicted = outputs.max(1)
                probs = F.softmax(outputs, dim=1)[:, 1]
                
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())
                
        # 7. Calculate Metrics
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(all_val_labels, all_val_preds) * 100
        val_precision = precision_score(all_val_labels, all_val_preds, zero_division=0) * 100
        val_recall = recall_score(all_val_labels, all_val_preds, zero_division=0) * 100
        
        try:
            val_auc = roc_auc_score(all_val_labels, all_val_probs)
        except ValueError:
            val_auc = float('nan')
            
        cm = confusion_matrix(all_val_labels, all_val_preds)
        
        # 8. Print and Log to W&B
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

        # 9. Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Optional: torch.save(model.state_dict(), "best_attention_probe.pt")
        else:
            patience_counter += 1
            print(f"Early Stopping Counter: {patience_counter} / {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered. Halting training.")
                break # Exit the epoch loop

if __name__ == "__main__":
    # Define the Hyperparameter Sweep Configuration
    sweep_config = {
        'method': 'bayes', # Bayesian optimization 
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'   
        },
        'parameters': {
            # 'batch_size': {'values':[16,32,64,128]},
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

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="orbis_attention_probe_40_data")

    # Run the sweep agent (this will run train_linear_probe 20 times with different parameters)
    wandb.agent(sweep_id, function=train_linear_probe, count=20)