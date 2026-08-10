import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import wandb
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_PROBES_DIR = PROJECT_ROOT / "DiagnosticProbes"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def resolve_path(p):
    if p is None:
        return None
    if os.path.isabs(p):
        return p
    p1 = os.path.join(DIAGNOSTIC_PROBES_DIR, p)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(PROJECT_ROOT, p)
    if os.path.exists(p2):
        return p2
    if os.path.exists(p):
        return os.path.abspath(p)
    return p1

try:
    from dota import DOTA_CLASS_NAMES
except ImportError:
    try:
        from DiagnosticProbes.scripts.dota import DOTA_CLASS_NAMES
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

import random
import numpy as np

def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
set_seed(43)


class CachedFeatureDataset(Dataset):
    def __init__(self, cache_path):
        resolved_cache_path = resolve_path(cache_path)
        data = torch.load(resolved_cache_path, map_location='cpu')
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


# =====================================================================
# 1. ORIGINAL SINGLE-TASK ATTENTION PROBE ARCHITECTURE
# =====================================================================
class AttentionProbe(nn.Module):
    def __init__(self, input_dim, num_classes=NUM_CLASS, num_heads=8):
        super().__init__()
        # Learnable query token ("Detective")
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.classifier = nn.Linear(input_dim, num_classes)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        x = self.norm1(x)
        attn_out, attn_weights = self.attn(query=q, key=x, value=x, average_attn_weights=False)
        pooled_features = attn_out.squeeze(1)
        pooled_features = self.norm2(pooled_features)
        logits = self.classifier(pooled_features)
        return logits, attn_weights


# =====================================================================
# 2. ALTERNATIVE MULTI-TASK ATTENTION PROBE ARCHITECTURE
# =====================================================================
class MultiTaskAttentionProbe(nn.Module):
    """
    Multi-Task Attention Probe with Dual Heads:
    - Shared Cross-Attention Query Backbone: Extracts spatial-temporal context
    - Head 1 (Binary Head): Classifies Normal (0) vs Anomaly (1)
    - Head 2 (Multiclass Head): Classifies fine-grained 10 categories
    """
    def __init__(self, input_dim, num_classes=NUM_CLASS, num_heads=8, dropout=0.2):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(p=dropout)

        # Dual Task Classification Heads
        self.binary_classifier = nn.Linear(input_dim, 2)          # Normal vs Anomaly
        self.mc_classifier = nn.Linear(input_dim, num_classes)    # 10 Multiclass Categories

    def forward(self, x):
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        x = self.norm1(x)
        attn_out, attn_weights = self.attn(query=q, key=x, value=x, average_attn_weights=False)
        pooled_features = attn_out.squeeze(1)
        pooled_features = self.norm2(pooled_features)
        pooled_features = self.dropout(pooled_features)

        binary_logits = self.binary_classifier(pooled_features)
        mc_logits = self.mc_classifier(pooled_features)
        return binary_logits, mc_logits, attn_weights


# =====================================================================
# 3. ALTERNATIVE MULTI-TASK LOSS FUNCTION
# =====================================================================
class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss (Cui et al., CVPR 2019):
    Re-weights multiclass loss using the effective number of samples:
    E_n = (1 - beta^N_c) / (1 - beta)
    Weight W_c = (1 - beta) / (1 - beta^N_c)
    """
    def __init__(self, samples_per_cls, num_classes=NUM_CLASS, beta=0.999):
        super().__init__()
        if samples_per_cls is not None:
            class_counts = torch.tensor(samples_per_cls, dtype=torch.float32)
            class_counts = torch.where(class_counts == 0, torch.tensor(1.0), class_counts)
            effective_num = (1.0 - torch.pow(beta, class_counts)) / (1.0 - beta)
            weights = (1.0 - beta) / effective_num
            weights = weights / weights.sum() * num_classes
            self.weights = weights
        else:
            self.weights = None

    def forward(self, logits, labels):
        device = logits.device
        weights = self.weights.to(device) if self.weights is not None else None
        return F.cross_entropy(logits, labels, weight=weights)


class MultiTaskLoss(nn.Module):
    """
    Multi-Task Loss combining:
    - Binary Loss: Standard Cross-Entropy (Normal vs Anomaly) [Kept Intact]
    - Multiclass Loss: Class-Balanced Loss (Cui et al., CVPR 2019)
    Loss_total = binary_loss_weight * Loss_binary + 1.0 * Loss_multiclass_CB
    """
    def __init__(self, samples_per_cls=None, binary_loss_weight=1.0, beta=0.999):
        super().__init__()
        self.binary_criterion = nn.CrossEntropyLoss()
        self.mc_criterion = ClassBalancedLoss(samples_per_cls=samples_per_cls, num_classes=NUM_CLASS, beta=beta)
        self.binary_loss_weight = binary_loss_weight

    def forward(self, binary_logits, mc_logits, binary_labels, mc_labels):
        loss_bin = self.binary_criterion(binary_logits, binary_labels)
        loss_mc = self.mc_criterion(mc_logits, mc_labels)
        total_loss = self.binary_loss_weight * loss_bin + 1.0 * loss_mc
        return total_loss, loss_bin, loss_mc


def train_linear_probe(use_wandb=False):
    # Hardcoded Hyperparameters from grateful-sweep-2:
    mode = 'multitask'               # 'single' or 'multitask'
    batch_size = 128
    learning_rate = 2.0481589106932835e-05
    weight_decay = 0.07288216277017416
    beta1 = 0.95
    beta2 = 0.99
    early_stopping_patience = 5
    binary_loss_weight = 1.834112626052137   # Tunable hyperparameter for binary loss weight (multiclass loss weight fixed to 1.0)
    cb_beta = 0.99                          # Tunable hyperparameter for Class-Balanced Loss effective number beta

    if use_wandb:
        wandb.init()
        config = wandb.config
        mode = getattr(config, 'mode', mode)
        batch_size = getattr(config, 'batch_size', batch_size)
        learning_rate = getattr(config, 'learning_rate', learning_rate)
        weight_decay = getattr(config, 'weight_decay', weight_decay)
        beta1 = getattr(config, 'beta1', beta1)
        beta2 = getattr(config, 'beta2', beta2)
        early_stopping_patience = getattr(config, 'early_stopping_patience', early_stopping_patience)
        binary_loss_weight = getattr(config, 'binary_loss_weight', binary_loss_weight)
        cb_beta = getattr(config, 'cb_beta', cb_beta)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"  Training Probe Mode: '{mode.upper()}' (binary_loss_weight={binary_loss_weight}, mc_loss_weight=1.0)")
    print(f"=======================================================")
    print(f"Using device: {device}")

    # Load Cached Data
    train_dataset = CachedFeatureDataset("./cached_features/train_block18_all_correct_unpooled_mc.pt")
    val_dataset = CachedFeatureDataset("./cached_features/val_block18_all_correct_unpooled_mc.pt")
    print(f'------- Train: {len(train_dataset)} | Val: {len(val_dataset)} ---------')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    hidden_dim = 768

    # Instantiate Model & Loss based on requested mode
    if mode == 'multitask':
        class_counts = torch.bincount(train_dataset.mc_labels.long(), minlength=NUM_CLASS).float()
        samples_per_cls = class_counts.cpu().numpy()

        cb_loss_module = ClassBalancedLoss(samples_per_cls=samples_per_cls, num_classes=NUM_CLASS, beta=cb_beta)
        print(f"\n--- Class-Balanced (CB) Multiclass Loss Weights (beta={cb_beta}) ---")
        for cls_id in range(NUM_CLASS):
            c_name = DOTA_CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
            cnt = int(samples_per_cls[cls_id])
            w = float(cb_loss_module.weights[cls_id].item())
            print(f"  Class {cls_id:2d} ({c_name:25s}): Count = {cnt:5d} | CB Weight = {w:.4f}")
        print("-" * 65 + "\n")

        model = MultiTaskAttentionProbe(input_dim=hidden_dim, num_classes=NUM_CLASS, num_heads=8).to(device)
        criterion = MultiTaskLoss(samples_per_cls=samples_per_cls, binary_loss_weight=binary_loss_weight, beta=cb_beta)
    else:
        # Original Single-Task Multiclass Attention Probe with Custom Split Weights
        if train_dataset.mc_labels is not None:
            class_counts = torch.bincount(train_dataset.mc_labels.long(), minlength=NUM_CLASS).float()
            class_counts = torch.where(class_counts == 0, torch.tensor(1.0), class_counts)

            class_weights = torch.zeros(NUM_CLASS, device=device)
            class_weights[0] = 0.5
            anomaly_inv_counts = 1.0 / class_counts[1:]
            anomaly_weights = anomaly_inv_counts / anomaly_inv_counts.sum()
            class_weights[1:] = 0.5 * anomaly_weights
        else:
            class_weights = None

        model = AttentionProbe(input_dim=hidden_dim, num_classes=NUM_CLASS, num_heads=8).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))

    best_val_loss = float('inf')
    patience = early_stopping_patience
    patience_counter = 0
    epochs = 50

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_bin_loss = 0.0
        total_train_mc_loss = 0.0
        correct_mc = 0
        correct_bin = 0
        total = 0

        for idx, batch_data in enumerate(train_loader):
            features = batch_data[0].to(device)
            binary_labels = batch_data[1].to(device)
            mc_labels = batch_data[2].to(device)

            optimizer.zero_grad()

            if mode == 'multitask':
                bin_logits, mc_logits, _ = model(features)
                loss, loss_bin, loss_mc = criterion(bin_logits, mc_logits, binary_labels, mc_labels)
                total_train_bin_loss += loss_bin.item()
                total_train_mc_loss += loss_mc.item()

                _, bin_pred = bin_logits.max(1)
                correct_bin += bin_pred.eq(binary_labels).sum().item()

                outputs = mc_logits
            else:
                outputs, _ = model(features)
                loss = criterion(outputs, mc_labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, mc_pred = outputs.max(1)
            total += mc_labels.size(0)
            correct_mc += mc_pred.eq(mc_labels).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc_mc = 100. * correct_mc / total
        train_acc_bin = (100. * correct_bin / total) if mode == 'multitask' else 0.0

        # Validation Loop
        model.eval()
        total_val_loss = 0.0
        total_val_bin_loss = 0.0
        total_val_mc_loss = 0.0

        all_val_mc_labels = []
        all_val_mc_preds = []
        all_val_mc_probs = []

        all_binary_labels = []
        all_binary_preds = []
        all_binary_probs = []

        current_epoch_attention_weights = {}

        with torch.no_grad():
            for batch_data in val_loader:
                features = batch_data[0].to(device)
                binary_labels = batch_data[1].to(device)
                mc_labels = batch_data[2].to(device)
                source_mc_labels = batch_data[3] if len(batch_data) > 3 else None
                video_ids = batch_data[5] if len(batch_data) > 5 else batch_data[3]
                target_frame_ids = batch_data[6] if len(batch_data) > 6 else [None] * len(video_ids)

                if mode == 'multitask':
                    bin_logits, mc_logits, attn_wts = model(features)
                    val_loss, val_bin_loss, val_mc_loss = criterion(bin_logits, mc_logits, binary_labels, mc_labels)
                    total_val_bin_loss += val_bin_loss.item()
                    total_val_mc_loss += val_mc_loss.item()

                    bin_probs = F.softmax(bin_logits, dim=1)[:, 1]
                    _, bin_preds = bin_logits.max(1)

                    outputs = mc_logits
                else:
                    outputs, attn_wts = model(features)
                    val_loss = criterion(outputs, mc_labels)
                    bin_probs = None
                    bin_preds = None

                total_val_loss += val_loss.item()

                _, mc_preds = outputs.max(1)
                mc_probs = F.softmax(outputs, dim=1)

                all_val_mc_labels.extend(mc_labels.cpu().numpy())
                all_val_mc_preds.extend(mc_preds.cpu().numpy())
                all_val_mc_probs.extend(mc_probs.cpu().numpy())

                all_binary_labels.extend(binary_labels.cpu().numpy())
                if mode == 'multitask':
                    all_binary_probs.extend(bin_probs.cpu().numpy())
                    all_binary_preds.extend(bin_preds.cpu().numpy())
                else:
                    all_binary_probs.extend((mc_probs[:, 1:].sum(dim=1)).cpu().numpy())

                if not use_wandb:
                    for i, id in enumerate(video_ids):
                        bin_lbl = int(binary_labels[i].item())
                        mc_id = int(mc_labels[i].item())
                        src_id = int(source_mc_labels[i].item()) if isinstance(source_mc_labels, torch.Tensor) and source_mc_labels[i].item() >= 0 else mc_id
                        class_label = DOTA_CLASS_NAMES.get(mc_id, f"Class_{mc_id}")
                        source_class_label = DOTA_CLASS_NAMES.get(src_id, f"Class_{src_id}") if src_id >= 0 else class_label

                        pred_label = int(mc_preds[i].item())
                        prob_mc = float(mc_probs[i, mc_id].item())
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

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc_mc = accuracy_score(all_val_mc_labels, all_val_mc_preds) * 100
        val_precision = precision_score(all_val_mc_labels, all_val_mc_preds, zero_division=0, average='macro') * 100
        val_recall = recall_score(all_val_mc_labels, all_val_mc_preds, zero_division=0, average='macro') * 100
        val_f1 = f1_score(all_val_mc_labels, all_val_mc_preds, zero_division=0, average='macro') * 100

        try:
            val_auc_mc = roc_auc_score(all_val_mc_labels, all_val_mc_probs, multi_class='ovr', average='macro')
            binary_auc = roc_auc_score(all_binary_labels, all_binary_probs)
        except ValueError:
            val_auc_mc = float('nan')
            binary_auc = float('nan')

        print(f"\nEpoch {epoch+1}/{epochs} [{mode.upper()}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"--> MC Val Acc: {val_acc_mc:.2f}% | Macro MC AUC: {val_auc_mc:.4f} | Macro F1: {val_f1:.2f}% | Binary AUC: {binary_auc:.4f}")

        if wandb.run is not None:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_accuracy_mc": train_acc_mc,
                "val_loss": avg_val_loss,
                "val_accuracy_mc": val_acc_mc,
                "val_precision_macro": val_precision,
                "val_recall_macro": val_recall,
                "val_f1_macro": val_f1,
                "val_auc_macro": val_auc_mc,
                "val_auc_binary": binary_auc,
                "mode": mode,
            }
            if mode == 'multitask':
                log_dict.update({
                    "train_loss_bin": total_train_bin_loss / len(train_loader),
                    "train_loss_mc": total_train_mc_loss / len(train_loader),
                    "val_loss_bin": total_val_bin_loss / len(val_loader),
                    "val_loss_mc": total_val_mc_loss / len(val_loader),
                    "train_accuracy_bin": train_acc_bin,
                })
            wandb.log(log_dict)

        # Early Stopping & Model Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if not use_wandb:
                checkpoint_dir = resolve_path("checkpoints/multiclass")
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt_name = f"best_multiclass_attention_probe_{mode}.pt"
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, ckpt_name))

                checkpoint_data = {"sequences": current_epoch_attention_weights}
                weights_name = f"best_multiclass_val_attention_weights_{mode}.pt"
                torch.save(checkpoint_data, os.path.join(checkpoint_dir, weights_name))
                print(f">>> Saved best model ({ckpt_name}) and weights ({weights_name}) to '{checkpoint_dir}' <<<")
        else:
            patience_counter += 1
            print(f"Early Stopping Counter: {patience_counter} / {patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered for mode '{mode}'. Halting training.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multiclass or Multi-Task Attention Probe.")
    parser.add_argument("--sweep", action="store_true", help="Enable W&B HPO hyperparameter sweep mode")
    parser.add_argument("--sweep_count", type=int, default=20, help="Number of sweep trials to run in W&B sweep mode")
    parser.add_argument("--project", type=str, default="orbis-attention-probe-mc-multitask", help="W&B Project name for sweep")
    args = parser.parse_args()

    if args.sweep:
        # Hyperparameter Sweep Configuration for W&B
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
                    'values': [64, 128]
                },
                'binary_loss_weight': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 2.0
                },
                'cb_beta': {
                    'values': [0.99, 0.999]
                },
                'beta1': {
                    'value': 0.95
                },
                'beta2': {
                    'value': 0.99
                },
                'early_stopping_patience': {
                    'value': 5
                },
                'mode': {
                    'value': 'multitask'
                }
            }
        }

        print(f"\n=======================================================")
        print(f" Launching W&B HPO Sweep Mode (project='{args.project}', trials={args.sweep_count})")
        print(f"=======================================================\n")
        sweep_id = wandb.sweep(sweep_config, project=args.project)
        wandb.agent(sweep_id, function=lambda: train_linear_probe(use_wandb=True), count=args.sweep_count)
    else:
        print("\n=======================================================")
        print(" Running Single Setting Test Mode (without W&B)")
        print("=======================================================\n")
        train_linear_probe(use_wandb=False)
