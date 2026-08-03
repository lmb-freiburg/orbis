import os
import sys
import argparse
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

DOTA_NAME_TO_ID = {v: k for k, v in DOTA_CLASS_NAMES.items()}

class CachedSurpriseDataset(Dataset):
    def __init__(self, cache_path="./cached_features/cached_raw_surprise_scores.pt", split='train', map_type='combined'):
        if not os.path.exists(cache_path):
            alt_path = os.path.join("..", cache_path)
            if os.path.exists(alt_path):
                cache_path = alt_path

        data = torch.load(cache_path, map_location='cpu')
        split_items = [d for d in data if d.get('split') == split]
        
        feats_list = []
        labels_list = []
        mc_labels_list = []
        src_mc_labels_list = []
        video_ids_list = []
        target_frame_ids_list = []

        for item in split_items:
            hm = item['head_maps'][map_type]
            if not isinstance(hm, torch.Tensor):
                hm = torch.tensor(hm)

            # hm shape: [T=4, C, H=18, W=32]
            # Take mean across time dimension T=4 (dim=0): [C, 18, 32]
            mean_hm = hm.float().mean(dim=0)
            # Permute to [18, 32, C] -> Reshape to [576 spatial tokens, C channels]
            feat = mean_hm.permute(1, 2, 0).reshape(576, -1)

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

        print(f"Loaded {cache_path} [{split} | map_type: '{map_type}'] | Feature Shape: {self.features.shape}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        mc = self.mc_labels[idx] if self.mc_labels is not None else -1
        src_mc = self.source_mc_labels[idx] if self.source_mc_labels is not None else -1
        ego = -1
        tf_id = self.target_frame_ids[idx] if idx < len(self.target_frame_ids) else None
        return self.features[idx].float(), self.labels[idx], mc, src_mc, ego, self.video_ids[idx], tf_id

class MaxPoolProbe(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x shape: [Batch, 576, input_dim]
        x = self.norm1(x)
        pooled_features = x.max(dim=1).values  # [Batch, input_dim]
        pooled_features = self.norm2(pooled_features)
        logits = self.classifier(pooled_features)
        return logits, None

def train_linear_probe(map_type='combined'):
    batch_size = 64
    learning_rate = 8.921907952045913e-05
    weight_decay = 0.018889857643545577
    beta1 = 0.95
    beta2 = 0.99
    early_stopping_patience = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"   Training MaxPool Probe for Head Map: '{map_type}'")
    print(f"=======================================================")
    print(f"Using device: {device}")

    train_dataset = CachedSurpriseDataset("./cached_features/cached_raw_surprise_scores.pt", split='train', map_type=map_type)
    val_dataset = CachedSurpriseDataset("./cached_features/cached_raw_surprise_scores.pt", split='val', map_type=map_type)

    print(f'------- Train: {len(train_dataset)} | Val: {len(val_dataset)} ---------')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    hidden_dim = train_dataset.features.shape[2]  # 16 for detailed/semantic, 32 for combined
    model = MaxPoolProbe(input_dim=hidden_dim, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2)
    )

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_auc = 0.0
    patience = early_stopping_patience
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

        with torch.no_grad():
            for batch_data in val_loader:
                features = batch_data[0].to(device)
                labels = batch_data[1].to(device)

                outputs, _ = model(features)
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()

                _, predicted = outputs.max(1)
                probs = F.softmax(outputs, dim=1)[:, 1]

                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(all_val_labels, all_val_preds) * 100

        try:
            val_auc = roc_auc_score(all_val_labels, all_val_probs)
        except ValueError:
            val_auc = float('nan')

        print(f"Epoch {epoch+1}/{epochs} [{map_type}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"--> Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | AUC: {val_auc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_val_auc = val_auc
            patience_counter = 0
            checkpoint_dir = "./checkpoints/surprise"
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"best_surprise_maxpool_probe_{map_type}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f">>> Saved new best model to '{ckpt_path}'! <<<")
        else:
            patience_counter += 1
            print(f"Early Stopping Counter: {patience_counter} / {patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered for '{map_type}'. Halting training.")
                break

    print(f"\n--- Best Eval Metrics for MaxPool Probe [{map_type}] ---")
    print(f"Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc:.2f}% | Best Val AUC: {best_val_auc:.4f}\n")
    return {"map_type": map_type, "val_loss": best_val_loss, "val_acc": best_val_acc, "val_auc": best_val_auc}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MaxPool Probe on raw surprise scores per head map.")
    parser.add_argument("--map_type", type=str, choices=['detailed', 'semantic', 'combined', 'all'], default='all',
                        help="Head map key to train on (detailed, semantic, combined, or all)")
    args = parser.parse_args()

    head_maps = ['detailed', 'semantic', 'combined'] if args.map_type == 'all' else [args.map_type]
    results = []
    for h_map in head_maps:
        res = train_linear_probe(map_type=h_map)
        results.append(res)

    print("\n=======================================================")
    print("        SUMMARY OF BEST EVAL POINTS (MAXPOOL PROBE)")
    print("=======================================================")
    for r in results:
        print(f"  Head Map: {r['map_type']:<10} | Best Val Loss: {r['val_loss']:.4f} | Best Val Acc: {r['val_acc']:.2f}% | Best Val AUC: {r['val_auc']:.4f}")
    print("=======================================================\n")
