import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import wandb # Added Weights & Biases

NUM_CLASS = 9

class CachedFeatureDataset(Dataset):
    def __init__(self, cache_path):
        data = torch.load(cache_path)
        self.features = data['features']
        self.labels = data['labels'].long() 
        self.mc_labels = data['mc_labels'] if 'mc_labels' in data else None
        self.video_ids = data['video_ids']
        # If you saved IDs, you can also load self.ids = data['ids'] here
        print(f'Loaded {cache_path} | Shape - {self.features.shape} , {self.labels.shape}')
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Add self.ids[idx] here if you added IDs to your caching script!
        if self.mc_labels is None:
            return self.features[idx], self.labels[idx], self.video_ids[idx]
        return self.features[idx], self.labels[idx], self.mc_labels[idx], self.video_ids[idx]

class AttentionProbe(nn.Module):
    def __init__(self, input_dim, num_classes=2, num_heads=8):
        super().__init__()
        # 1. Learnable query token (Think of this as the "Detective")
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # 2. Multi-head Attention Layer
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        
        # 3. Final linear classifier
        self.classifier = nn.Linear(input_dim, num_classes)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x is your unpooled cached features. Shape: [Batch, 576, 768]
        B = x.size(0)
        
        # Expand our single query token to match the batch size
        q = self.query.expand(B, -1, -1)

        # Norm 1 Application
        x = self.norm1(x)
        
        # Cross-Attention
        attn_out, attn_weights = self.attn(query=q, key=x, value=x)

        
        # Squeeze out the sequence dimension: [Batch, 1, 768] -> [Batch, 768]
        pooled_features = attn_out.squeeze(1)
        
        #Norm 2
        pooled_features = self.norm2(pooled_features)

        # Pass the attended features into the final classifier
        logits = self.classifier(pooled_features)
        
        return logits, attn_weights

def train_linear_probe():
    # 1. Initialize W&B run (Config is populated by the Sweep agent)
    wandb.init()
    config = wandb.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load Cached Data from Block 18
    train_dataset = CachedFeatureDataset("./cached_features/train_block18_unpooled_mc.pt")
    val_dataset = CachedFeatureDataset("./cached_features/val_block18_unpooled_mc.pt")

    print(f'------- Train: {len(train_dataset)} | Val: {len(val_dataset)} ---------')
    
    # Drop last to avoid batch size mismatch issues in edge cases
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 3. Initialize Attention Probe
    hidden_dim = 768 
    num_classes = NUM_CLASS
    model = AttentionProbe(input_dim=hidden_dim, num_heads=8, num_classes=num_classes).to(device)
    
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
    
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct = 0
        total = 0
        for idx, batch_data in enumerate(train_loader):
            features = batch_data[0].to(device)
            # labels = batch_data[1].to(device)
            mc_labels = batch_data[2].to(device)
            video_ids = batch_data[-1]
            
            optimizer.zero_grad()
            outputs, _ = model(features)
            loss = criterion(outputs, mc_labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += mc_labels.size(0)
            correct += predicted.eq(mc_labels).sum().item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # 5. Validation Loop
        model.eval()
        total_val_loss = 0
        
        all_val_mc_labels = []
        all_val_preds = []
        all_val_probs = []

        current_epoch_attention_weights = {}

        with torch.no_grad():
            for batch_data in val_loader:
                features = batch_data[0].to(device)
                # labels = batch_data[1].to(device)
                mc_labels = batch_data[2].to(device)
                video_ids = batch_data[-1]
                outputs, attn_wts = model(features)
                
                # Calculate Validation Loss for Early Stopping
                val_loss = criterion(outputs, mc_labels)
                total_val_loss += val_loss.item()
                
                _, predicted = outputs.max(1)
                probs = F.softmax(outputs, dim=1)
                
                all_val_mc_labels.extend(mc_labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())

                # Extract and save attention map per sequence ID directly to files
                for i, id in enumerate(video_ids):
                    current_epoch_attention_weights[id] = attn_wts[i].squeeze(0).cpu()
                
        # 6. Calculate Metrics
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(all_val_mc_labels, all_val_preds) * 100
        val_precision = precision_score(all_val_mc_labels, all_val_preds, zero_division=0, average='weighted') * 100
        val_recall = recall_score(all_val_mc_labels, all_val_preds, zero_division=0, average='weighted') * 100
        
        try:
            val_auc = roc_auc_score(all_val_mc_labels, all_val_probs, multi_class='ovr')
        except ValueError:
            val_auc = float('nan')
            
        cm = confusion_matrix(all_val_mc_labels, all_val_preds)
        
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
            #Save the model state
            torch.save(model.state_dict(), "best_attention_probe.pt")
            
            # Save the attention weights for the validation set of this best epoch
            torch.save(current_epoch_attention_weights, "best_val_attention_weights.pt")
            print(">>> Saved new best model and attention weights! <<<")
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





    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="orbis-attention-probe-mc")
    # Run the sweep agent (this will run train_linear_probe 20 times with different parameters)
    wandb.agent(sweep_id, function=train_linear_probe, count=20)


    # # # Load your attention weights look-up map
    # attn_map = torch.load("best_val_attention_weights.pt")

    # # Query weights for a specific target sequence ID
    # my_sequence_id = "sequence_xyz_123" 
    # weights = attn_map[my_sequence_id] # Tensors shape: [576]

    # # Reshape back to spatial token map dimension (e.g., 24x24 if 576 tokens)
    # spatial_weights = weights.reshape(18, 32)


