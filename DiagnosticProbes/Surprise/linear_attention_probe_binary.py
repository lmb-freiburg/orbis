import os
import sys
import argparse
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import wandb

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

DOTA_NAME_TO_ID = {v: k for k, v in DOTA_CLASS_NAMES.items()}

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

class CachedSurpriseDataset(Dataset):
    def __init__(self, cache_path="./cached_features/cached_normalized_surprise_scores_3000.pt", split='train', map_type='combined', t_step='mean'):
        resolved_cache_path = resolve_path(cache_path)
        if not resolved_cache_path or not os.path.exists(resolved_cache_path):
            for alt in ["cached_features/cached_normalized_surprise_scores_3000.pt", "results/sample_scores.pt"]:
                alt_res = resolve_path(alt)
                if alt_res and os.path.exists(alt_res):
                    resolved_cache_path = alt_res
                    break

        data = torch.load(resolved_cache_path, map_location='cpu')
        split_items = [d for d in data if d.get('split') == split]
        
        feats_list = []
        labels_list = []
        mc_labels_list = []
        src_mc_labels_list = []
        video_ids_list = []
        target_frame_ids_list = []

        half = 16

        for item in split_items:
            # Always load 'combined' as it contains both semantic (first 16 channels) and detailed (last 16 channels)
            hm = item['head_maps']['combined']
            if not isinstance(hm, torch.Tensor):
                hm = torch.tensor(hm)

            # hm shape: [T=4, C=32, H=18, W=32]
            #Semantic is 2nd half and 1st is Detailed
            if map_type == 'semantic':
                hm = hm[:, half:, :, :]
            elif map_type == 'detailed':
                hm = hm[:, :half, :, :]

            if t_step == 'mean' or t_step is None:
                # Mean across time dimension T=4 (dim=0): [C, 18, 32]
                selected_hm = hm.float().mean(dim=0)
            else:
                t_idx = int(t_step)
                # Select specific time step t: [C, 18, 32]
                selected_hm = hm[t_idx].float()

            # Permute to [18, 32, C] -> Reshape to [576 spatial tokens, C channels]
            feat = selected_hm.permute(1, 2, 0).reshape(576, -1)

            feats_list.append(feat)
            labels_list.append(item.get('label', 0))

            acc_name = item.get('accident_name', 'normal')
            src_acc_name = item.get('source_accident_name', acc_name)
            mc_id = DOTA_NAME_TO_ID.get(acc_name, 0 if acc_name == 'normal' else -1)
            src_mc_id = DOTA_NAME_TO_ID.get(src_acc_name, mc_id)

            mc_labels_list.append(mc_id)
            src_mc_labels_list.append(src_mc_id)
            video_ids_list.append(item.get('clip_id', ''))
            target_frame_ids_list.append(item.get('target_frame_id', None))

        self.features = torch.stack(feats_list)
        self.labels = torch.tensor(labels_list, dtype=torch.long)
        self.mc_labels = torch.tensor(mc_labels_list, dtype=torch.long)
        self.source_mc_labels = torch.tensor(src_mc_labels_list, dtype=torch.long)
        self.ego_labels = None
        self.video_ids = video_ids_list
        self.target_frame_ids = target_frame_ids_list

        print(f"Loaded {cache_path} [{split} | map_type: '{map_type}' | t_step: '{t_step}'] | Feature Shape: {self.features.shape}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        mc = self.mc_labels[idx] if self.mc_labels is not None else -1
        src_mc = self.source_mc_labels[idx] if self.source_mc_labels is not None else -1
        ego = -1
        tf_id = self.target_frame_ids[idx] if idx < len(self.target_frame_ids) else None
        return self.features[idx].float(), self.labels[idx], mc, src_mc, ego, self.video_ids[idx], tf_id

class AttentionProbe(nn.Module):
    def __init__(self, input_dim, num_classes=2, num_heads=4):
        super().__init__()
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

def train_linear_probe(map_type='combined', t_step='3', use_wandb=False):
    # Hardcoded Hyperparameters from new best run:
    batch_size = 64
    learning_rate = 0.0001591540688380618
    weight_decay = 0.04198262672171282
    beta1 = 0.95
    beta2 = 0.999
    early_stopping_patience = 5

    if use_wandb:
        wandb.init()
        config = wandb.config
        batch_size = config.batch_size
        learning_rate = config.learning_rate
        weight_decay = config.weight_decay
        beta1 = getattr(config, 'beta1', beta1)
        beta2 = getattr(config, 'beta2', beta2)
        early_stopping_patience = getattr(config, 'early_stopping_patience', early_stopping_patience)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    mode_title = "W&B Sweep" if use_wandb else "Local Run (sweet-sweep-2)"
    print(f"\n=======================================================")
    print(f"  [{mode_title}] Probe for Head Map: '{map_type}' | t_step: '{t_step}'")
    print(f"=======================================================")
    print(f"Using device: {device}")

    train_dataset = CachedSurpriseDataset("./cached_features/cached_normalized_surprise_scores_3000.pt", split='train', map_type=map_type, t_step=t_step)
    val_dataset = CachedSurpriseDataset("./cached_features/cached_normalized_surprise_scores_3000.pt", split='val', map_type=map_type, t_step=t_step)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    hidden_dim = train_dataset.features.shape[2]  # 16 for detailed/semantic, 32 for combined
    num_heads = 4
    model = AttentionProbe(input_dim=hidden_dim, num_classes=2, num_heads=num_heads).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_auc = 0.0
    patience_counter = 0
    epochs = 50

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct = 0
        total = 0
        for idx, batch_data in enumerate(train_loader):
            features = batch_data[0].to(device)
            labels = batch_data[1].to(device)

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
                mc_labels = batch_data[2]
                source_mc_labels = batch_data[3]
                video_ids = batch_data[5]
                target_frame_ids = batch_data[6]

                outputs, attn_wts = model(features)
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()

                _, predicted = outputs.max(1)
                probs = F.softmax(outputs, dim=1)[:, 1]

                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())

                if not use_wandb:
                    for i, id in enumerate(video_ids):
                        bin_lbl = int(labels[i].item())
                        mc_id = int(mc_labels[i].item()) if isinstance(mc_labels, torch.Tensor) else int(mc_labels[i])
                        src_id = int(source_mc_labels[i].item()) if isinstance(source_mc_labels, torch.Tensor) and source_mc_labels[i].item() >= 0 else mc_id
                        class_label = DOTA_CLASS_NAMES.get(mc_id, f"Class_{mc_id}")
                        source_class_label = DOTA_CLASS_NAMES.get(src_id, f"Class_{src_id}") if src_id >= 0 else class_label

                        pred_label = int(predicted[i].item())
                        prob_val = float(probs[i].item())
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
                            "prob_true": prob_val
                        }

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(all_val_labels, all_val_preds) * 100
        val_precision = precision_score(all_val_labels, all_val_preds, zero_division=0) * 100
        val_recall = recall_score(all_val_labels, all_val_preds, zero_division=0) * 100

        try:
            val_auc = roc_auc_score(all_val_labels, all_val_probs)
        except ValueError:
            val_auc = float('nan')

        print(f"Epoch {epoch+1}/{epochs} [{map_type} | t={t_step}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | AUC: {val_auc:.4f}")

        if use_wandb and wandb.run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_accuracy": train_acc,
                "val_loss": avg_val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_auc": val_auc,
                "head_map": map_type,
                "t_step": str(t_step)
            })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_val_auc = val_auc
            patience_counter = 0
            if not use_wandb:
                checkpoint_dir = resolve_path("checkpoints/surprise")
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt_name = f"best_binary_attention_probe_{map_type}_t{t_step}.pt"
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, ckpt_name))

                checkpoint_data = {"sequences": current_epoch_attention_weights}
                weights_name = f"best_binary_val_attention_weights_{map_type}_t{t_step}.pt"
                torch.save(checkpoint_data, os.path.join(checkpoint_dir, weights_name))
                print(f">>> Saved best model ({ckpt_name}) and weights ({weights_name}) to '{checkpoint_dir}' <<<")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered for '{map_type}' (t={t_step}). Halting training.")
                break

    print(f"\n--- Best Eval Metrics for [{map_type} | t={t_step}] ---")
    print(f"Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc:.2f}% | Best Val AUC: {best_val_auc:.4f}\n")
    return {"map_type": map_type, "t_step": str(t_step), "val_loss": best_val_loss, "val_acc": best_val_acc, "val_auc": best_val_auc}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Attention Probe on normalized surprise scores.")
    parser.add_argument("--map_type", type=str, choices=['detailed', 'semantic', 'combined', 'all'], default='all',
                        help="Head map key to train on (detailed, semantic, combined, or all)")
    parser.add_argument("--t_step", type=str, choices=['0', '1', '2', '3', 'mean', 'all', 'extremes'], default='extremes',
                        help="Time step index to train on (0, 1, 2, 3, mean, or all)")
    parser.add_argument("--sweep", action="store_true", help="Enable W&B HPO hyperparameter sweep mode")
    parser.add_argument("--sweep_count", type=int, default=10,
                        help="Number of Bayesian hyperparameter sweep runs per map_type and t_step")
    args = parser.parse_args()

    head_maps = ['detailed', 'semantic', 'combined'] if args.map_type == 'all' else [args.map_type]
    if args.t_step == 'extremes':
        t_steps = ['0', '3']
    elif args.t_step == 'all':
        t_steps = [0, 1, 2, 3, 'mean']
    elif args.t_step.isdigit():
        t_steps = [int(args.t_step)]
    else:
        t_steps = args.t_step

    if args.sweep:
        sweep_config = {
            'method': 'bayes',
            'metric': {'name': 'val_loss', 'goal': 'minimize'},
            'parameters': {
                'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
                'weight_decay': {'distribution': 'uniform', 'min': 0.0, 'max': 0.1},
                'batch_size': {'values': [16, 32, 64]},
                'beta1': {'values': [0.9, 0.95]},
                'beta2': {'values': [0.99, 0.999]},
                'early_stopping_patience': {'value': 5}
            }
        }

        for h_map in head_maps:
            for t_step in t_steps:
                project_name = f"orbis-surprise-attention-probe-{h_map}_t"
                print(f"\n=======================================================")
                print(f" Launching W&B Sweep for Project: '{project_name}' (t={t_step})")
                print(f"=======================================================")
                sweep_id = wandb.sweep(sweep_config, project=project_name)
                train_fn = functools.partial(train_linear_probe, map_type=h_map, t_step=t_step, use_wandb=True)
                wandb.agent(sweep_id, function=train_fn, count=args.sweep_count)
    else:
        for h_map in head_maps:
            for t_step in t_steps:
                print(f"\n=======================================================")
                print(f" Running Local Single Run (sweet-sweep-2 parameters): '{h_map}' (t={t_step})")
                print(f"=======================================================")
                train_linear_probe(map_type=h_map, t_step=t_step, use_wandb=False)
