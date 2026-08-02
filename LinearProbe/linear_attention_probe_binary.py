import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import wandb # Added Weights & Biases

try:
    from dota import DOTA_CLASS_NAMES
except ImportError:
    try:
        from LinearProbe.dota import DOTA_CLASS_NAMES
    except ImportError:
        DOTA_CLASS_NAMES = {
            0: "normal",
            1: "start_stop_or_stationary",
            2: "moving_ahead_or_waiting",
            3: "lateral",
            4: "oncoming",
            5: "turning",
            6: "pedestrian",
            7: "obstacle",
            8: "leave_to_right",
            9: "leave_to_left",
            10: "unknown",
        }

class CachedFeatureDataset(Dataset):
    def __init__(self, cache_path):
        data = torch.load(cache_path, map_location='cpu')
        self.features = data['features'].half() if data['features'].dtype == torch.float32 else data['features']
        self.labels = data['labels'].long() 
        self.mc_labels = data['mc_labels'] if 'mc_labels' in data else None
        self.source_mc_labels = data['source_mc_labels'] if 'source_mc_labels' in data else None
        self.ego_labels = data['ego_labels'] if 'ego_labels' in data else None
        self.video_ids = data['video_ids']
        self.target_frame_ids = data['target_frame_ids'] if 'target_frame_ids' in data else [None] * len(self.video_ids)
        print(f'Loaded {cache_path} | Shape - {self.features.shape} , {self.labels.shape}')
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        mc = self.mc_labels[idx] if self.mc_labels is not None else -1
        src_mc = self.source_mc_labels[idx] if self.source_mc_labels is not None else -1
        ego = self.ego_labels[idx] if self.ego_labels is not None else -1
        tf_id = self.target_frame_ids[idx] if idx < len(self.target_frame_ids) else None
        return self.features[idx].float(), self.labels[idx], mc, src_mc, ego, self.video_ids[idx], tf_id

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
        attn_out, attn_weights = self.attn(query=q, key=x, value=x, average_attn_weights=False)

        
        # Squeeze out the sequence dimension: [Batch, 1, 768] -> [Batch, 768]
        pooled_features = attn_out.squeeze(1)
        
        #Norm 2
        pooled_features = self.norm2(pooled_features)

        # Pass the attended features into the final classifier
        logits = self.classifier(pooled_features)
        
        return logits, attn_weights

def train_linear_probe():
    wandb.init()
    config = wandb.config

    # Dynamic W&B Sweep hyperparameter configuration
    batch_size = getattr(config, 'batch_size', 32)
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    beta1 = config.beta1
    beta2 = config.beta2
    early_stopping_patience = 5

    # Labeled best hyperparameters setting (commented out for sweep HPO):
    # batch_size = 32
    # learning_rate = 0.00001935262894891232
    # weight_decay = 0.056775318543155096
    # beta1 = 0.95
    # beta2 = 0.99
    # early_stopping_patience = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load Cached Data from Block 18
    train_dataset = CachedFeatureDataset("./cached_features/train_block18_3600_correct_unpooled_mc.pt")
    val_dataset = CachedFeatureDataset("./cached_features/val_block18_3600_correct_unpooled_mc.pt")

    print(f'------- Train: {len(train_dataset)} | Val: {len(val_dataset)} ---------')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Initialize Attention Probe
    hidden_dim = 768 
    model = AttentionProbe(input_dim=hidden_dim, num_heads=8).to(device)

    criterion = nn.CrossEntropyLoss()
    # Use HPO configs for AdamW
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay,
        betas=(beta1, beta2)
    )

    # Early Stopping Setup
    best_val_loss = float('inf')
    patience = early_stopping_patience
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
            labels = batch_data[1].to(device)
            video_ids = batch_data[-1]

            optimizer.zero_grad()
            outputs, _ = model(features)
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
        total_val_loss = 0.0
        all_val_labels = []
        all_val_preds = []
        all_val_probs = []
        
        current_epoch_attention_weights = {}

        with torch.no_grad():
            for batch_data in val_loader:
                features = batch_data[0].to(device)
                labels = batch_data[1].to(device)
                if len(batch_data) == 7:
                    mc_labels = batch_data[2]
                    source_mc_labels = batch_data[3]
                    ego_labels = batch_data[4]
                    video_ids = batch_data[5]
                    target_frame_ids = batch_data[6]
                elif len(batch_data) == 6:
                    mc_labels = batch_data[2]
                    source_mc_labels = batch_data[3]
                    ego_labels = None
                    video_ids = batch_data[4]
                    target_frame_ids = batch_data[5]
                elif len(batch_data) == 5:
                    mc_labels = batch_data[2]
                    source_mc_labels = None
                    ego_labels = None
                    video_ids = batch_data[3]
                    target_frame_ids = batch_data[4]
                else:
                    mc_labels = None
                    source_mc_labels = None
                    ego_labels = None
                    video_ids = batch_data[2]
                    target_frame_ids = [None] * len(video_ids)

                outputs, attn_wts = model(features)
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()
                
                _, predicted = outputs.max(1)
                probs = F.softmax(outputs, dim=1)[:, 1]
                
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())

                for i, id in enumerate(video_ids):
                    binary_label = int(labels[i].item())
                    # Target class label for multiclass (0 for normal, 1..9 for anomaly)
                    mc_id = int(mc_labels[i].item()) if isinstance(mc_labels, torch.Tensor) else -1
                    # Source sequence category label (1..9 for both normal and anomaly clips from sequence)
                    src_id = int(source_mc_labels[i].item()) if isinstance(source_mc_labels, torch.Tensor) and source_mc_labels[i].item() >= 0 else mc_id
                    
                    class_id = mc_id if mc_id >= 0 else binary_label
                    class_label = DOTA_CLASS_NAMES.get(class_id, f"Class_{class_id}")
                    source_class_label = DOTA_CLASS_NAMES.get(src_id, f"Class_{src_id}") if src_id >= 0 else class_label
                    
                    pred_label = int(predicted[i].item())
                    prob_anom = float(probs[i].item())
                    prob_true = prob_anom if binary_label == 1 else (1.0 - prob_anom)
                    target_frame_id = target_frame_ids[i] if i < len(target_frame_ids) else None
                    unique_key = f"{id}_{target_frame_id}" if target_frame_id else f"{id}_lbl{binary_label}"

                    current_epoch_attention_weights[unique_key] = {
                        "video_id": id,
                        "attn_weights": attn_wts[i].squeeze(1).cpu(),
                        "target_frame_id": target_frame_id,
                        "class_id": class_id,
                        "class_label": class_label,
                        "source_class_id": src_id,
                        "source_class_label": source_class_label,
                        "binary_label": binary_label,
                        "pred_label": pred_label,
                        "prob_anom": prob_anom,
                        "prob_true": prob_true
                    }
        
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
        
        # 7. Print Console Output (wandb logging temporarily commented out)
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
            # # Save the model state
            # torch.save(model.state_dict(), "best_attention_probe.pt")

            # # Identify Worst Mistakes (lowest prob_true)
            # fps = [
            #     (vid, info) for vid, info in current_epoch_attention_weights.items()
            #     if info["binary_label"] == 0 and info["pred_label"] == 1
            # ]
            # fns = [
            #     (vid, info) for vid, info in current_epoch_attention_weights.items()
            #     if info["binary_label"] == 1 and info["pred_label"] == 0
            # ]

            # fps.sort(key=lambda x: x[1]["prob_true"])
            # fns.sort(key=lambda x: x[1]["prob_true"])

            # fp_ids = [vid for vid, _ in fps]
            # fn_ids = [vid for vid, _ in fns]

            # # Save the attention weights and pre-sorted FP/FN IDs for the validation set of this best epoch
            # checkpoint_data = {
            #     "sequences": current_epoch_attention_weights,
            #     "fps": fp_ids,
            #     "fns": fn_ids
            # }
            # torch.save(checkpoint_data, "best_val_attention_weights.pt")
            # print(">>> Saved new best model and attention weights with FP/FN IDs! <<<")

            # print("\n--- WORST MISTAKES SUMMARY ---")
            # print("Top False Positives (Normal predicted as Anomalous with high confidence):")
            # for vid, info in fps[:5]:
            #     print(f"  - Video: {vid} | P(Anomalous): {info['prob_anom']:.4f} | True Class: {info['class_label']} (ID: {info['class_id']})")
            # print("Top False Negatives (Anomalous predicted as Normal with high confidence):")
            # for vid, info in fns[:5]:
            #     print(f"  - Video: {vid} | P(Anomalous): {info['prob_anom']:.4f} | True Class: {info['class_label']} (ID: {info['class_id']})")
            # print("-------------------------------\n")
        else:
            patience_counter += 1
            print(f"Early Stopping Counter: {patience_counter} / {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered. Halting training.")
                break


if __name__ == "__main__":

    sweep_config = {
        'method': 'bayes', 
        'metric': {'name': 'val_loss', 'goal': 'minimize'},
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

    # Comment out best hyperparam setting
    # sweep_config = {
    # 'method': 'grid',  # Changed to grid to run this specific configuration once
    # 'metric': {
    #     'name': 'val_loss',
    #     'goal': 'minimize'   
    # },
    # 'parameters': {
    #     'learning_rate': {
    #         'value': 0.00001935262894891232  # Exact best learning rate
    #     },
    #     'weight_decay': {
    #         'value': 0.056775318543155096    # Exact best weight decay
    #     },
    #     'batch_size': {
    #         'value': 32
    #     },
    #     'beta1': {
    #         'value': 0.95
    #     },
    #     'beta2': {
    #         'value': 0.99
    #     },
    #     'early_stopping_patience': {
    #         'value': 5
    #     }
    # }
    # }
    # # Best Hyper Param-Setting - dauntless-sweep-15
    #     batch_size:32
    #     beta1:0.95
    #     beta2:0.99
    #     early_stopping_patience:5
    #     learning_rate:0.00001935262894891232
    #     weight_decay:0.056775318543155096

    # # Best Summary metrics for above Hyperparameters setting
    # {
    #     "_step": 33,
    #     "epoch": 34,
    #     "_wandb.runtime": 98,
    #     "val_auc": 0.7982466971231016,
    #     "_runtime": 98,
    #     "val_loss": 0.5442327558994293,
    #     "_timestamp": 1784202666.857404,
    #     "train_loss": 0.4139233272184025,
    #     "val_recall": 63.73626373626373,
    #     "val_accuracy": 69.44444444444444,
    #     "val_precision": 72.5,
    #     "train_accuracy": 81.5340909090909
    # }



    # Initialize the sweep with project name containing '-corrected-3600'
    sweep_id = wandb.sweep(sweep_config, project="orbis-attention-probe-weights-corrected-3600")
    # Run the sweep agent across 20 experiments
    wandb.agent(sweep_id, function=train_linear_probe, count=20)

    # # # Load your attention weights look-up map
    # attn_map = torch.load("best_val_attention_weights.pt")

    # # Query weights for a specific target sequence ID
    # my_sequence_id = "sequence_xyz_123" 
    # weights = attn_map[my_sequence_id] # Tensors shape: [576]

    # # Reshape back to spatial token map dimension (e.g., 24x24 if 576 tokens)
    # spatial_weights = weights.reshape(18, 32)


