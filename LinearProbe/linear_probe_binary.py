import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import wandb # Added Weights & Biases

class CachedFeatureDataset(Dataset):
    def __init__(self, cache_path):
        data = torch.load(cache_path)
        self.features = data['features']
        self.labels = data['labels'].long() 
        print(f'Loaded {cache_path} | Shape - {self.features.shape}')
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        return self.classifier(self.norm(x))

def train_linear_probe():
    # 1. Initialize W&B run (Config is populated by the Sweep agent)
    wandb.init()
    config = wandb.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Cached Data
    train_dataset = CachedFeatureDataset("./cached_features/train_block18.pt")
    val_dataset = CachedFeatureDataset("./cached_features/val_block18.pt")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 3. Initialize Probe

    # Maxpooling
    hidden_dim = 768 

    # UnPooled - Using the flat dimension from your previous setup
    # hidden_dim = 576 * 768 


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
    epochs = 50 # Max epochs, but early stopping will likely cut this short
    
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct = 0
        total = 0
        
        for idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            # Flattening the activation
            B, H, W = features.shape
            #UnPooled
            # features = features.reshape(B, H*W)
            #Maxpooled
            features = torch.max(features, dim=1)[0]

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
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)

                # Flattening the activation
                B, H, W = features.shape
                #Unpooled
                # features = features.reshape(B, H*W)
                #Maxpooled
                features = torch.max(features, dim=1)[0]


                outputs = model(features)
                
                # Calculate Validation Loss for Early Stopping
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
            
        cm = confusion_matrix(all_val_labels, all_val_preds)
        
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            checkpoint_dir = "./checkpoints/binary"
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_binary_linear_probe.pt"))
            print(f">>> Saved new best binary linear model to '{checkpoint_dir}'! <<<")
        else:
            patience_counter += 1
            print(f"Early Stopping Counter: {patience_counter} / {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered. Halting training.")
                break # Exit the epoch loop

if __name__ == "__main__":
    # Define the Hyperparameter Sweep Configuration
    # sweep_config = {
    #     'method': 'bayes', # Bayesian optimization (finds the best params faster than random)
    #     'metric': {
    #         'name': 'val_loss',
    #         'goal': 'minimize'   
    #     },
    #     'parameters': {
    #         'learning_rate': {
    #             'distribution': 'log_uniform_values',
    #             'min': 1e-6,
    #             'max': 1e-3
    #         },
    #         'weight_decay': {
    #             'distribution': 'uniform',
    #             'min': 0.0,
    #             'max': 0.1
    #         },
    #         'batch_size': {
    #             'values': [16, 32, 64]
    #         },
    #         'beta1': {
    #             'values': [0.9, 0.95]
    #         },
    #         'beta2': {
    #             'values': [0.99, 0.999]
    #         },
    #         'early_stopping_patience': {
    #             'value': 5
    #         }
    #     }
    # }
    sweep_config = {
    'method': 'grid',  # Changed to grid since we are running a single fixed point
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'   
    },
    'parameters': {
        'learning_rate': {
            'value': 0.0006432805604895261  # Exact best learning rate
        },
        'weight_decay': {
            'value': 0.06599430117405974    # Exact best weight decay
        },
        'batch_size': {
            'value': 64
        },
        'beta1': {
            'value': 0.95
        },
        'beta2': {
            'value': 0.999
        },
        'early_stopping_patience': {
            'value': 5
        }
    }
    }
    # # Best Hyper Param-Setting - smooth-sweep-20
    # batch_size:64
    # beta1:0.95
    # beta2:0.999
    # early_stopping_patience:5
    # learning_rate:0.0006432805604895261
    # weight_decay:0.06599430117405974

    # # Best Summary metrics for above Hyperparameters setting
    # {
    #     "_step": 49,
    #     "epoch": 50,
    #     "_wandb.runtime": 16,
    #     "val_auc": 0.7210766761328559,
    #     "_runtime": 16,
    #     "val_loss": 0.6219724218050638,
    #     "_timestamp": 1784200511.4264178,
    #     "train_loss": 0.5774389157692591,
    #     "val_recall": 52.74725274725275,
    #     "val_accuracy": 66.11111111111111,
    #     "val_precision": 72.72727272727273,
    #     "train_accuracy": 72.5
    # }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="orbis-linear-maxpool-probe")

    # Run the sweep agent (this will run train_linear_probe 20 times with different parameters)
    wandb.agent(sweep_id, function=train_linear_probe, count=20)