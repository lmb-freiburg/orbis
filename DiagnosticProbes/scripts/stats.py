import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_PROBES_DIR = PROJECT_ROOT / "DiagnosticProbes"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def resolve_path(p):
    if p is None:
        return None
    p_path = Path(p)
    if p_path.is_absolute():
        return str(p_path)
    p1 = PROJECT_ROOT / p
    if p1.exists():
        return str(p1)
    p2 = DIAGNOSTIC_PROBES_DIR / p
    if p2.exists():
        return str(p2)
    return str(p1)

from DiagnosticProbes.scripts.dota import DOTA_CLASS_NAMES
from DiagnosticProbes.Activations.linear_attention_probe_binary import AttentionProbe
DOTA_NAME_TO_ID = {v: k for k, v in DOTA_CLASS_NAMES.items()}


class CachedFeatureDataset(Dataset):
    def __init__(self, cache_path, split='val', map_type='combined', t_step='3'):
        cache_path = resolve_path(cache_path)
        data = torch.load(cache_path, map_location='cpu')
        if isinstance(data, list):
            split_items = [d for d in data if d.get('split') == split]
            feats_list, labels_list, mc_labels_list, src_mc_labels_list, video_ids_list, target_frame_ids_list = [], [], [], [], [], []
            half = 16
            for item in split_items:
                hm = item['head_maps']['combined']
                if not isinstance(hm, torch.Tensor):
                    hm = torch.tensor(hm)
                if map_type == 'semantic':
                    hm = hm[:, half:, :, :]
                elif map_type == 'detailed':
                    hm = hm[:, :half, :, :]

                if t_step == 'mean' or t_step is None:
                    selected_hm = hm.float().mean(dim=0)
                else:
                    t_idx = int(t_step)
                    selected_hm = hm[t_idx].float()

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
            self.video_ids = video_ids_list
            self.target_frame_ids = target_frame_ids_list
        else:
            self.features = data['features'].half() if data['features'].dtype == torch.float32 else data['features']
            self.labels = data['labels'].long()
            self.mc_labels = data['mc_labels'] if 'mc_labels' in data else None
            self.source_mc_labels = data['source_mc_labels'] if 'source_mc_labels' in data else None
            self.video_ids = data['video_ids']
            self.target_frame_ids = data['target_frame_ids'] if 'target_frame_ids' in data else [None] * len(self.video_ids)
        print(f"Loaded cached features from '{cache_path}' | Features shape: {self.features.shape}, Labels shape: {self.labels.shape}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        mc = self.mc_labels[idx] if self.mc_labels is not None else -1
        src_mc = self.source_mc_labels[idx] if self.source_mc_labels is not None else -1
        tf_id = self.target_frame_ids[idx] if idx < len(self.target_frame_ids) else None
        return self.features[idx].float(), self.labels[idx], mc, src_mc, self.video_ids[idx], tf_id


class AttentionProbe(nn.Module):
    def __init__(self, input_dim, num_classes=2, num_heads=8):
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


class MultiTaskAttentionProbe(nn.Module):
    def __init__(self, input_dim, num_classes=10, num_heads=8, dropout=0.2):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.binary_classifier = nn.Linear(input_dim, 2)
        self.mc_classifier = nn.Linear(input_dim, num_classes)

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


def load_probe_model(model_path, weights_path, input_dim, num_classes, device, is_mc=False, num_heads=None):
    """
    Instantiates AttentionProbe or MultiTaskAttentionProbe and loads state_dict.
    """
    if num_heads is None:
        num_heads = 4 if (model_path and "surprise" in model_path) else 8

    model_path = resolve_path(model_path)
    weights_path = resolve_path(weights_path)

    candidate_model_paths = [
        model_path,
        os.path.join(PROJECT_ROOT, "checkpoints", "multiclass", "best_multiclass_attention_probe_multitask.pt"),
        os.path.join(PROJECT_ROOT, "checkpoints", "multiclass", "best_multiclass_attention_probe_single.pt"),
        os.path.join(PROJECT_ROOT, "checkpoints", "multiclass", "best_multiclass_attention_probe.pt"),
        os.path.join(PROJECT_ROOT, "checkpoints", "binary", "best_binary_attention_probe.pt"),
    ]

    for path in candidate_model_paths:
        if path and os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location=device)
                if isinstance(state_dict, dict):
                    if "mc_classifier.weight" in state_dict or "binary_classifier.weight" in state_dict:
                        model = MultiTaskAttentionProbe(input_dim=input_dim, num_classes=num_classes, num_heads=num_heads).to(device)
                        model.load_state_dict(state_dict)
                        print(f"Successfully loaded MultiTaskAttentionProbe (num_heads={num_heads}) weights from '{path}'")
                        model.eval()
                        return model, True
                    elif "classifier.weight" in state_dict or "query" in state_dict:
                        model = AttentionProbe(input_dim=input_dim, num_classes=num_classes, num_heads=num_heads).to(device)
                        model.load_state_dict(state_dict)
                        print(f"Successfully loaded AttentionProbe (num_heads={num_heads}) weights from '{path}'")
                        model.eval()
                        return model, False
            except Exception as e:
                print(f"Warning: Could not load weights from '{path}': {e}")

    print(f"Warning: No trained probe weights found! Running with newly initialized AttentionProbe weights (num_heads={num_heads}).")
    model = AttentionProbe(input_dim=input_dim, num_classes=num_classes, num_heads=num_heads).to(device)
    model.eval()
    return model, False


DOTA_CLASS_CODES = {
    0: "NO",
    1: "SS",
    2: "MW",
    3: "LA",
    4: "OC",
    5: "TU",
    6: "PD",
    7: "OB",
    8: "LR",
    9: "LL",
}


def plot_10x10_confusion_matrix(all_labels, all_preds, output_dir="./confusion_matrices_mc"):
    """
    Generates a 10x10 color-coded confusion matrix heatmap for all 10 DoTA classes.
    Axes: Y-axis = True Label (Full name + code in brackets), X-axis = Predicted Label (2-letter code).
    """
    output_dir = resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    from sklearn.metrics import confusion_matrix
    
    formatted_class_names = [
        "normal",
        "start_stop_or_\nstationary",
        "moving_ahead_\nor_waiting",
        "lateral",
        "oncoming",
        "turning",
        "pedestrian",
        "obstacle",
        "leave_to_right",
        "leave_to_left"
    ]
    y_class_names = [f"{formatted_class_names[i]} ({DOTA_CLASS_CODES.get(i, 'XX')})" for i in range(10)]
    x_class_names = [DOTA_CLASS_CODES.get(i, 'XX') for i in range(10)]
    cm_10x10 = confusion_matrix(all_labels, all_preds, labels=list(range(10)))
    
    row_sums = cm_10x10.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_10x10.astype('float'), row_sums, out=np.zeros_like(cm_10x10, dtype=float), where=row_sums!=0)

    fig, ax = plt.subplots(figsize=(16, 12))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        ax=ax,
        linewidths=1,
        linecolor="white",
        xticklabels=x_class_names,
        yticklabels=y_class_names,
        annot_kws={"weight": "bold", "size": 13}
    )

    ax.set_xlabel("Predicted Label", fontsize=16, fontweight='bold', labelpad=12)
    ax.set_ylabel("True Label", fontsize=16, fontweight='bold', labelpad=12)
    ax.set_title("10x10 Multiclass Confusion Matrix Heatmap (True Label vs Predicted Label)", fontsize=18, fontweight='bold', pad=15)
    plt.xticks(rotation=0, ha='center', fontsize=14, fontweight='bold')
    plt.yticks(rotation=0, fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, "cm_10x10_multiclass_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 10x10 Multiclass Confusion Matrix Heatmap plot to '{save_path}'")
    return save_path


def save_confusion_matrix_plots(class_grouped, overall_stats, output_dir="./confusion_matrices_binary", is_mc=False):
    """
    Generates high-resolution color-coded confusion matrix plots for each source_class_label
    as well as the overall validation variant, and a combined 2x5 multi-panel summary grid image.
    """
    output_dir = resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    all_classes = sorted(class_grouped.keys())

    # Items to plot: Each source class + Overall
    items = list(all_classes) + ['OVERALL']

    # 1. Combined Summary Grid Plot (2 rows x 5 cols)
    fig, axes = plt.subplots(2, 5, figsize=(26, 11))
    axes = axes.flatten()

    variant_title = "Multiclass (MC)" if is_mc else "Binary"
    plt.suptitle(f"Color-Coded Confusion Matrices Per Source Class & Overall ({variant_title})", fontsize=20, fontweight='bold', y=0.98)

    for idx, item in enumerate(items):
        ax = axes[idx]
        if item == 'OVERALL':
            tn, fp, fn, tp = overall_stats['tn'], overall_stats['fp'], overall_stats['fn'], overall_stats['tp']
            title = f"OVERALL VALIDATION (N={overall_stats['total']})\nAcc: {overall_stats['acc']:.1f}% | AUC: {overall_stats['auc_str']}"
            single_filename = "cm_overall_mc.png" if is_mc else "cm_overall.png"
        else:
            c_data = class_grouped[item]
            c_name = DOTA_CLASS_NAMES.get(item, f"Class_{item}")
            t_labels = c_data["true_labels"]
            p_labels = c_data["pred_labels"]
            
            if is_mc:
                # For MC evaluation per source category: true category vs predicted category
                tn = sum(1 for y, p in zip(t_labels, p_labels) if y == p)
                fp = sum(1 for y, p in zip(t_labels, p_labels) if y != p)
                fn = 0
                tp = 0
            else:
                tn = sum(1 for y, p in zip(t_labels, p_labels) if y == 0 and p == 0)
                fp = sum(1 for y, p in zip(t_labels, p_labels) if y == 0 and p == 1)
                fn = sum(1 for y, p in zip(t_labels, p_labels) if y == 1 and p == 0)
                tp = sum(1 for y, p in zip(t_labels, p_labels) if y == 1 and p == 1)
                
            total = len(t_labels)
            acc = (tp + tn) / total * 100 if total > 0 else 0.0
            title = f"Class {item}: {c_name} (N={total})\nAcc: {acc:.1f}%"
            single_filename = f"cm_class_{item}_{c_name}_mc.png" if is_mc else f"cm_class_{item}_{c_name}.png"

        cm = np.array([[tn, fp], [fn, tp]])
        n_norm = tn + fp
        n_anom = fn + tp

        tn_pct = f"{tn/n_norm*100:.1f}%" if n_norm > 0 else "0%"
        fp_pct = f"{fp/n_norm*100:.1f}%" if n_norm > 0 else "0%"
        fn_pct = f"{fn/n_anom*100:.1f}%" if n_anom > 0 else "0%"
        tp_pct = f"{tp/n_anom*100:.1f}%" if n_anom > 0 else "0%"

        text_annotations = [
            [f"TN/Corr: {tn}\n({tn_pct})", f"FP/Err: {fp}\n({fp_pct})"],
            [f"FN: {fn}\n({fn_pct})", f"TP: {tp}\n({tp_pct})"]
        ]

        # Draw Heatmap on Subplot
        sns.heatmap(cm, annot=False, fmt="d", cmap="YlGnBu", cbar=False, ax=ax, linewidths=2, linecolor='white')

        for r in range(2):
            for c in range(2):
                cell_color = "red" if (r != c and cm[r, c] > 0) else "darkgreen" if (r == c and cm[r, c] > 0) else "black"
                ax.text(c + 0.5, r + 0.5, text_annotations[r][c],
                        ha="center", va="center", color=cell_color, fontsize=12, fontweight='bold')

        ax.set_xticklabels(["Pred Normal (0)", "Pred Anom (1)"] if not is_mc else ["Correct", "Error"], fontsize=10, fontweight='bold')
        ax.set_yticklabels(["True Normal (0)", "True Anom (1)"] if not is_mc else ["Target", "Other"], fontsize=10, fontweight='bold', rotation=0)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)

        # 2. Save Individual High-Res Confusion Matrix Plot
        fig_single, ax_single = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=False, fmt="d", cmap="YlGnBu", cbar=False, ax=ax_single, linewidths=3, linecolor='white')
        for r in range(2):
            for c in range(2):
                cell_color = "#b71c1c" if (r != c and cm[r, c] > 0) else "#1b5e20" if (r == c and cm[r, c] > 0) else "#212121"
                ax_single.text(c + 0.5, r + 0.5, text_annotations[r][c],
                               ha="center", va="center", color=cell_color, fontsize=16, fontweight='bold')

        ax_single.set_xticklabels(["Pred Normal (0)", "Pred Anom (1)"] if not is_mc else ["Correct", "Error"], fontsize=11, fontweight='bold')
        ax_single.set_yticklabels(["True Normal (0)", "True Anom (1)"] if not is_mc else ["Target", "Other"], fontsize=11, fontweight='bold', rotation=0)
        ax_single.set_title(title, fontsize=14, fontweight='bold', pad=12)
        plt.tight_layout()
        single_path = os.path.join(output_dir, single_filename)
        plt.savefig(single_path, dpi=200, bbox_inches='tight')
        plt.close(fig_single)

    # Hide any unused subplots
    for idx in range(len(items), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    grid_filename = "cm_all_classes_grid_mc.png" if is_mc else "cm_all_classes_grid.png"
    grid_path = os.path.join(output_dir, grid_filename)
    plt.savefig(grid_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined color-coded confusion matrix grid plot to '{grid_path}'")
    print(f"Saved individual color-coded confusion matrix plots to '{output_dir}/'")

    return grid_path


def generate_stats(
    cache_path="./cached_features/val_block18_3600_correct_unpooled_mc.pt",
    model_path="best_attention_probe.pt",
    weights_path="best_val_attention_weights.pt",
    output_path="DiagnosticProbes/reports/stats_report_binary.md",
    plot_dir="DiagnosticProbes/confusionMatrices/binary",
    heatmap_dir="DiagnosticProbes/heatmaps/binary",
    batch_size=32,
    is_mc=False
):
    cache_path = resolve_path(cache_path)
    model_path = resolve_path(model_path)
    weights_path = resolve_path(weights_path)
    output_path = resolve_path(output_path)
    plot_dir = resolve_path(plot_dir)
    heatmap_dir = resolve_path(heatmap_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device} | Mode: {'Multiclass (MC)' if is_mc else 'Binary'}")

    if not os.path.exists(cache_path):
        print(f"Error: Cache file '{cache_path}' does not exist!")
        return

    dataset = CachedFeatureDataset(cache_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    input_dim = dataset.features.shape[-1]
    num_classes = 10 if is_mc else 2
    model, is_multitask = load_probe_model(model_path, weights_path, input_dim, num_classes, device, is_mc=is_mc)

    all_labels = []
    all_preds = []
    all_probs = []
    all_source_mc = []
    all_mc = []
    all_video_ids = []

    print(f"\nRunning forward pass through {'Multiclass' if is_mc else 'Binary'} Attention Probe...")
    with torch.no_grad():
        for batch in dataloader:
            features = batch[0].to(device)
            binary_labels = batch[1].to(device)
            mc_labels = batch[2].to(device) if is_mc else batch[2]
            source_mc_labels = batch[3]
            vids = batch[4]

            if is_multitask:
                bin_logits, mc_logits, _ = model(features)
                logits = mc_logits if is_mc else bin_logits
            else:
                logits, _ = model(features)

            probs = F.softmax(logits, dim=1)
            _, preds = logits.max(1)

            target_labels = mc_labels if is_mc else binary_labels

            all_labels.extend(target_labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_source_mc.extend(source_mc_labels.numpy())
            all_mc.extend(batch[2].numpy() if isinstance(batch[2], torch.Tensor) else batch[2])
            all_video_ids.extend(vids)

    total_samples = len(all_labels)
    print(f"Completed evaluation on {total_samples} validation samples.\n")

    # Group by source_class_label
    class_grouped = {}
    for idx in range(total_samples):
        src_id = int(all_source_mc[idx])
        mc_id = int(all_mc[idx])
        if src_id < 0:
            src_id = mc_id if mc_id >= 0 else int(all_labels[idx])

        if src_id not in class_grouped:
            class_grouped[src_id] = {
                "true_labels": [],
                "pred_labels": [],
                "probs": [],
                "video_ids": []
            }

        class_grouped[src_id]["true_labels"].append(all_labels[idx])
        class_grouped[src_id]["pred_labels"].append(all_preds[idx])
        class_grouped[src_id]["probs"].append(all_probs[idx])
        class_grouped[src_id]["video_ids"].append(all_video_ids[idx])

    # Overall Metrics across all validation data
    if is_mc:
        overall_tn = sum(1 for y, p in zip(all_labels, all_preds) if y == p)
        overall_fp = sum(1 for y, p in zip(all_labels, all_preds) if y != p)
        overall_fn = 0
        overall_tp = 0
        overall_acc = accuracy_score(all_labels, all_preds) * 100
        overall_prec = precision_score(all_labels, all_preds, zero_division=0, average='macro') * 100
        overall_rec = recall_score(all_labels, all_preds, zero_division=0, average='macro') * 100
        overall_f1 = f1_score(all_labels, all_preds, zero_division=0, average='macro') * 100
        try:
            overall_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
            overall_auc_str = f"{overall_auc:.4f}"
        except Exception:
            overall_auc_str = "N/A"
    else:
        overall_tn = sum(1 for y, p in zip(all_labels, all_preds) if y == 0 and p == 0)
        overall_fp = sum(1 for y, p in zip(all_labels, all_preds) if y == 0 and p == 1)
        overall_fn = sum(1 for y, p in zip(all_labels, all_preds) if y == 1 and p == 0)
        overall_tp = sum(1 for y, p in zip(all_labels, all_preds) if y == 1 and p == 1)

        overall_acc = accuracy_score(all_labels, all_preds) * 100
        overall_prec = precision_score(all_labels, all_preds, zero_division=0) * 100
        overall_rec = recall_score(all_labels, all_preds, zero_division=0) * 100
        overall_f1 = f1_score(all_labels, all_preds, zero_division=0) * 100
        try:
            val_probs_binary = [p[1] if len(p) > 1 else p[0] for p in all_probs]
            overall_auc = roc_auc_score(all_labels, val_probs_binary)
            overall_auc_str = f"{overall_auc:.4f}"
        except Exception:
            overall_auc_str = "N/A"

    overall_stats = {
        "total": total_samples,
        "tn": overall_tn,
        "fp": overall_fp,
        "fn": overall_fn,
        "tp": overall_tp,
        "acc": overall_acc,
        "prec": overall_prec,
        "rec": overall_rec,
        "f1": overall_f1,
        "auc_str": overall_auc_str
    }

    # Generate Color-Coded Confusion Matrix Plots (Grid + Individual PNGs)
    grid_plot_path = save_confusion_matrix_plots(class_grouped, overall_stats, output_dir=plot_dir, is_mc=is_mc)

    if is_mc:
        cm_10x10_path = plot_10x10_confusion_matrix(all_labels, all_preds, output_dir=plot_dir)

    # Generate Markdown Report
    report_lines = []
    report_lines.append(f"# {'Multiclass (MC)' if is_mc else 'Binary'} Attention Probe - Per Source Class Performance Report\n")
    report_lines.append(f"**Cached Features**: `{cache_path}`  ")
    report_lines.append(f"**Total Samples**: `{total_samples}`  \n")

    if is_mc:
        report_lines.append("## 1. 10x10 Multiclass Confusion Matrix Heatmap\n")
        report_lines.append(f"![10x10 Multiclass Confusion Matrix Heatmap]({plot_dir}/cm_10x10_multiclass_heatmap.png)\n\n---\n")

    report_lines.append("## 2. Summary Table Per Source Class (Accident Category)\n")
    header_table = "| Class ID | Source Class Name | Total | Target (1/MC) | Normal/Other | Correct | Error | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC |"
    divider_table = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    report_lines.append(header_table)
    report_lines.append(divider_table)

    print(header_table)
    print(divider_table)

    for src_id in sorted(class_grouped.keys()):
        c_data = class_grouped[src_id]
        t_labels = c_data["true_labels"]
        p_labels = c_data["pred_labels"]
        c_probs = c_data["probs"]

        c_name = DOTA_CLASS_NAMES.get(src_id, f"Class_{src_id}")
        n_total = len(t_labels)
        correct_cnt = sum(1 for y, p in zip(t_labels, p_labels) if y == p)
        error_cnt = n_total - correct_cnt

        if is_mc:
            target_cnt = sum(1 for y in t_labels if y == src_id)
            normal_cnt = n_total - target_cnt
        else:
            target_cnt = sum(1 for y in t_labels if y == 1)
            normal_cnt = sum(1 for y in t_labels if y == 0)

        acc = correct_cnt / n_total * 100 if n_total > 0 else 0.0
        prec = precision_score(t_labels, p_labels, zero_division=0, average='macro' if is_mc else 'binary') * 100
        rec = recall_score(t_labels, p_labels, zero_division=0, average='macro' if is_mc else 'binary') * 100
        f1 = f1_score(t_labels, p_labels, zero_division=0, average='macro' if is_mc else 'binary') * 100

        try:
            if is_mc:
                auc_str = f"{roc_auc_score(t_labels, c_probs, multi_class='ovr'):.4f}"
            else:
                c_probs_binary = [p[1] if len(p) > 1 else p[0] for p in c_probs]
                auc_str = f"{roc_auc_score(t_labels, c_probs_binary):.4f}"
        except Exception:
            auc_str = "N/A"

        row = f"| {src_id:<8} | {c_name:<25} | {n_total:<5} | {target_cnt:<13} | {normal_cnt:<12} | {correct_cnt:<7} | {error_cnt:<5} | {acc:<12.2f} | {prec:<13.2f} | {rec:<10.2f} | {f1:<12.2f} | {auc_str:<5} |"
        report_lines.append(row)
        print(row)

    if is_mc:
        overall_target = sum(1 for y in all_labels if y > 0)
        overall_normal = sum(1 for y in all_labels if y == 0)
        overall_correct = overall_tn
        overall_error = overall_fp
    else:
        overall_target = overall_fn + overall_tp
        overall_normal = overall_tn + overall_fp
        overall_correct = overall_tn + overall_tp
        overall_error = overall_fn + overall_fp

    overall_row = f"| **ALL**   | **OVERALL VALIDATION**    | {total_samples:<5} | {overall_target:<13} | {overall_normal:<12} | {overall_correct:<7} | {overall_error:<5} | {overall_acc:<12.2f} | {overall_prec:<13.2f} | {overall_rec:<10.2f} | {overall_f1:<12.2f} | {overall_auc_str:<5} |"
    report_lines.append(divider_table)
    report_lines.append(overall_row)
    print(divider_table)
    print(overall_row)

    grid_fig_name = "cm_all_classes_grid_mc.png" if is_mc else "cm_all_classes_grid.png"
    report_lines.append("\n---\n")
    report_lines.append("## 2. Combined Color-Coded Confusion Matrix Grid\n")
    report_lines.append(f"![All Classes Confusion Matrix Grid]({plot_dir}/{grid_fig_name})\n\n---\n")

    report_lines.append("## 3. Color-Coded Confusion Matrices Per Source Class\n")

    # Insert individual PNG plots
    for src_id in sorted(class_grouped.keys()):
        c_data = class_grouped[src_id]
        t_labels = c_data["true_labels"]
        p_labels = c_data["pred_labels"]
        c_name = DOTA_CLASS_NAMES.get(src_id, f"Class_{src_id}")
        n_total = len(t_labels)
        correct_cnt = sum(1 for y, p in zip(t_labels, p_labels) if y == p)
        acc = correct_cnt / n_total * 100 if n_total > 0 else 0.0

        single_img_name = f"cm_class_{src_id}_{c_name}_mc.png" if is_mc else f"cm_class_{src_id}_{c_name}.png"
        single_img_path = f"{plot_dir}/{single_img_name}"

        report_lines.append(f"### Class {src_id}: {c_name}")
        report_lines.append(f"**Total**: `{n_total}` | **Accuracy**: `{acc:.2f}%` | **Correct**: `{correct_cnt}` | **Error**: `{n_total - correct_cnt}`  \n")
        report_lines.append(f"![Confusion Matrix Class {src_id} - {c_name}]({single_img_path})\n\n---\n")

    overall_img_name = "cm_overall_mc.png" if is_mc else "cm_overall.png"
    report_lines.append("## 4. Overall Color-Coded Confusion Matrix (Entire Variant)\n")
    report_lines.append(f"**Total Validation Samples**: `{total_samples}` | **Overall Accuracy**: `{overall_acc:.2f}%` | **Overall AUC**: `{overall_auc_str}`  \n")
    report_lines.append(f"![Overall Confusion Matrix]({plot_dir}/{overall_img_name})\n")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nSaved detailed per-source-class performance report with color-coded matrices to '{output_path}'")

    try:
        from DiagnosticProbes.scripts.heatmaps import generate_attention_heatmaps_binary
    except Exception as e:
        print(f"Notice: Could not import heatmap generator: {e}")

    try:
        generate_attention_heatmaps_binary(
            weights_path=weights_path,
            sequence_dir=os.path.join(PROJECT_ROOT, "..", "DoTA_sequences"),
            output_dir=heatmap_dir
        )
    except Exception as e:
        print(f"Notice: Heatmap generation skipped or failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate per source_class_label stats & color-coded confusion matrices for Attention Probe.")
    parser.add_argument("--is_mc", action="store_true", help="Set flag for Multiclass evaluation (default: Binary)")
    parser.add_argument("--cache_path", type=str, default=None, help="Path to cached validation features")
    parser.add_argument("--model_path", type=str, default=None, help="Path to saved Attention Probe state_dict")
    parser.add_argument("--weights_path", type=str, default=None, help="Path to saved attention weights checkpoint")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save output Markdown report")
    parser.add_argument("--plot_dir", type=str, default=None, help="Directory to save confusion matrix plots")
    parser.add_argument("--heatmap_dir", type=str, default=None, help="Directory to save attention heatmaps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")

    args = parser.parse_args()

    # Set defaults based on is_mc flag
    if args.is_mc:
        cache_path = args.cache_path or "./cached_features/val_block18_all_correct_unpooled_mc.pt"
        model_path = args.model_path or "./checkpoints/multiclass/best_multiclass_attention_probe_multitask.pt"
        weights_path = args.weights_path or "./checkpoints/multiclass/best_multiclass_val_attention_weights_multitask.pt"
        output_path = args.output_path or "DiagnosticProbes/reports/stats_report_mc.md"
        plot_dir = args.plot_dir or "DiagnosticProbes/confusionMatrices/mc"
        heatmap_dir = args.heatmap_dir or "DiagnosticProbes/heatmaps/mc"
    else:
        cache_path = args.cache_path or "./cached_features/val_block18_all_correct_unpooled_mc.pt"
        model_path = args.model_path or "./checkpoints/binary/best_binary_attention_probe.pt"
        weights_path = args.weights_path or "./checkpoints/binary/best_binary_val_attention_weights.pt"
        output_path = args.output_path or "DiagnosticProbes/reports/stats_report_binary.md"
        plot_dir = args.plot_dir or "DiagnosticProbes/confusionMatrices/binary"
        heatmap_dir = args.heatmap_dir or "DiagnosticProbes/heatmaps/binary"

    generate_stats(
        cache_path=cache_path,
        model_path=model_path,
        weights_path=weights_path,
        output_path=output_path,
        plot_dir=plot_dir,
        heatmap_dir=heatmap_dir,
        batch_size=args.batch_size,
        is_mc=args.is_mc
    )
