import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import wandb

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

NUM_CLASS = 10


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
    def __init__(self, input_dim, num_classes=NUM_CLASS, num_heads=8):
        super().__init__()
        # 1. Learnable query token ("Detective")
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # 2. Multi-head Attention Layer
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        
        # 3. Final linear classifier
        self.classifier = nn.Linear(input_dim, num_classes)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x shape: [Batch, 576, 768]
        B = x.size(0)
        
        # Expand single query token to match batch size
        q = self.query.expand(B, -1, -1)
        
        # Norm 1 Application
        x = self.norm1(x)
        
        # Cross-Attention
        attn_out, attn_weights = self.attn(query=q, key=x, value=x, average_attn_weights=False)

        # Squeeze sequence dimension: [Batch, 1, 768] -> [Batch, 768]
        pooled_features = attn_out.squeeze(1)
        
        # Norm 2 Application
        pooled_features = self.norm2(pooled_features)

        # Pass attended features into final multiclass classifier
        logits = self.classifier(pooled_features)
        
        return logits, attn_weights


def train_linear_probe():
    # Local mode hyperparameters (skipping wandb)
    batch_size = 64
    learning_rate = 8.921907952045913e-05
    weight_decay = 0.018889857643545577
    beta1 = 0.95
    beta2 = 0.99
    early_stopping_patience = 5



    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load Cached Multiclass Data from Block 18
    train_dataset = CachedFeatureDataset("./cached_features/train_block18_3600_correct_unpooled_mc.pt")
    val_dataset = CachedFeatureDataset("./cached_features/val_block18_3600_correct_unpooled_mc.pt")

    print(f'------- Train: {len(train_dataset)} | Val: {len(val_dataset)} ---------')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Initialize Multiclass Attention Probe & Weighted Loss
    hidden_dim = 768 
    model = AttentionProbe(input_dim=hidden_dim, num_classes=NUM_CLASS, num_heads=8).to(device)

    # Compute class weights: Class 0 (normal) receives 0.5 of total loss weight mass,
    # while the remaining 0.5 is distributed inversely proportional to anomaly class counts.
    if train_dataset.mc_labels is not None:
        class_counts = torch.bincount(train_dataset.mc_labels.long(), minlength=NUM_CLASS).float()
        class_counts = torch.where(class_counts == 0, torch.tensor(1.0), class_counts)
        
        class_weights = torch.zeros(NUM_CLASS, device=device)
        class_weights[0] = 0.5
        
        anomaly_inv_counts = 1.0 / class_counts[1:]
        anomaly_weights = anomaly_inv_counts / anomaly_inv_counts.sum()
        class_weights[1:] = 0.5 * anomaly_weights
        print("\n--- Class Weights for Weighted CrossEntropyLoss ---")
        for cls_id in range(NUM_CLASS):
            c_name = DOTA_CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
            cnt = int(class_counts[cls_id].item())
            w = float(class_weights[cls_id].item())
            print(f"  Class {cls_id:2d} ({c_name:25s}): Count = {cnt:5d} | Weight = {w:.4f}")
        print("-" * 55 + "\n")
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
    epochs = 50

    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct = 0
        total = 0
        for idx, batch_data in enumerate(train_loader):
            features = batch_data[0].to(device)
            binary_labels = batch_data[1].to(device)
            mc_labels = batch_data[2].to(device)

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
        total_val_loss = 0.0
        all_val_mc_labels = []
        all_val_preds = []
        all_val_probs = []
        all_binary_labels = []
        all_binary_probs = []
        
        current_epoch_attention_weights = {}

        with torch.no_grad():
            for batch_data in val_loader:
                features = batch_data[0].to(device)
                binary_labels = batch_data[1].to(device)
                if len(batch_data) == 7:
                    mc_labels = batch_data[2].to(device)
                    source_mc_labels = batch_data[3]
                    ego_labels = batch_data[4]
                    video_ids = batch_data[5]
                    target_frame_ids = batch_data[6]
                elif len(batch_data) == 6:
                    mc_labels = batch_data[2].to(device)
                    source_mc_labels = batch_data[3]
                    ego_labels = None
                    video_ids = batch_data[4]
                    target_frame_ids = batch_data[5]
                elif len(batch_data) == 5:
                    mc_labels = batch_data[2].to(device)
                    source_mc_labels = None
                    ego_labels = None
                    video_ids = batch_data[3]
                    target_frame_ids = batch_data[4]
                else:
                    mc_labels = batch_data[1].to(device)
                    source_mc_labels = None
                    ego_labels = None
                    video_ids = batch_data[2]
                    target_frame_ids = [None] * len(video_ids)

                outputs, attn_wts = model(features)
                val_loss = criterion(outputs, mc_labels)
                total_val_loss += val_loss.item()
                
                _, predicted = outputs.max(1)
                probs = F.softmax(outputs, dim=1)
                
                all_val_mc_labels.extend(mc_labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())
                all_binary_labels.extend(binary_labels.cpu().numpy())
                all_binary_probs.extend((probs[:, 1:].sum(dim=1)).cpu().numpy())

                for i, id in enumerate(video_ids):
                    bin_lbl = int(binary_labels[i].item())
                    mc_id = int(mc_labels[i].item())
                    src_id = int(source_mc_labels[i].item()) if isinstance(source_mc_labels, torch.Tensor) and source_mc_labels[i].item() >= 0 else mc_id
                    
                    class_label = DOTA_CLASS_NAMES.get(mc_id, f"Class_{mc_id}")
                    source_class_label = DOTA_CLASS_NAMES.get(src_id, f"Class_{src_id}") if src_id >= 0 else class_label
                    
                    pred_label = int(predicted[i].item())
                    prob_mc = float(probs[i, mc_id].item())
                    target_frame_id = target_frame_ids[i] if i < len(target_frame_ids) else None
                    unique_key = f"{id}_{target_frame_id}" if target_frame_id else f"{id}_mc{mc_id}"

                    current_epoch_attention_weights[unique_key] = {
                        "video_id": id,
                        "attn_weights": attn_wts[i].squeeze(1).cpu(),
                        "target_frame_id": target_frame_id,
                        "class_id": mc_id,
                        "class_label": class_label,
                        "source_class_id": src_id,
                        "source_class_label": source_class_label,
                        "binary_label": bin_lbl,
                        "pred_label": pred_label,
                        "prob_true": prob_mc
                    }
        
        # 6. Calculate Metrics
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(all_val_mc_labels, all_val_preds) * 100
        val_precision = precision_score(all_val_mc_labels, all_val_preds, zero_division=0, average='weighted') * 100
        val_recall = recall_score(all_val_mc_labels, all_val_preds, zero_division=0, average='weighted') * 100
        
        try:
            val_auc = roc_auc_score(all_val_mc_labels, all_val_probs, multi_class='ovr', average='weighted')
            binary_auc = roc_auc_score(all_binary_labels, all_binary_probs)
        except ValueError:
            val_auc = float('nan')
            binary_auc = float('nan')
            
        cm = confusion_matrix(all_val_mc_labels, all_val_preds)
        
        # 7. Print Console Output
        print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"--> Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Weighted MC AUC: {val_auc:.4f} | Binary AUC: {binary_auc:.4f}")
        
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_accuracy": train_acc,
                "val_loss": avg_val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": (val_precision + val_recall) / 2,
                "val_auc_weighted": val_auc,
                "val_auc_binary": binary_auc,
                "class_weights": class_weights.tolist() if isinstance(class_weights, torch.Tensor) else None,
            })


        # 8. Early Stopping & Saving Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the model state
            checkpoint_dir = "./checkpoints/multiclass"
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_multiclass_attention_probe.pt"))

            # Save attention weights for best epoch
            checkpoint_data = {
                "sequences": current_epoch_attention_weights,
            }
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, "best_multiclass_val_attention_weights.pt"))
            print(f">>> Saved new best multiclass model and attention weights to '{checkpoint_dir}'! <<<")

            # print("\n--- WORST MISTAKES SUMMARY ---")
            # for vid, info in mistakes[:5]:
            #     print(f"  - Video: {vid} | True Class: {info['class_label']} (ID: {info['class_id']}) | Predicted: {DOTA_CLASS_NAMES.get(info['pred_label'])} (ID: {info['pred_label']}) | Prob(True): {info['prob_true']:.4f}")
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
                'value': 0.95
            },
            'beta2': {
                'value': 0.99
            },
            'early_stopping_patience': {
                'value': 5
            }
        }
    }

    # Initialize sweep (commented out for direct single run):
    # sweep_id = wandb.sweep(sweep_config, project="orbis-attention-probe-mc-corrected-3600")
    # wandb.agent(sweep_id, function=train_linear_probe, count=20)

    # Run single training & checkpoint saving using local best hyperparameters
    train_linear_probe()
