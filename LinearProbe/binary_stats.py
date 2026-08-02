import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
        data = torch.load(cache_path)
        self.features = data['features']
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
        return self.features[idx], self.labels[idx], mc, src_mc, self.video_ids[idx], tf_id


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


def load_probe_model(model_path, weights_path, input_dim, device):
    """
    Instantiates AttentionProbe and loads weights from model_path (state_dict)
    or weights_path if state_dict is present there.
    """
    model = AttentionProbe(input_dim=input_dim, num_classes=2, num_heads=8).to(device)

    loaded = False
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            if isinstance(state_dict, dict) and "query" in state_dict:
                model.load_state_dict(state_dict)
                print(f"Successfully loaded trained AttentionProbe weights from '{model_path}'")
                loaded = True
        except Exception as e:
            print(f"Warning: Could not load weights from '{model_path}': {e}")

    if not loaded and os.path.exists(weights_path):
        try:
            ckpt = torch.load(weights_path, map_location=device)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
                print(f"Successfully loaded trained weights from '{weights_path}'")
                loaded = True
        except Exception as e:
            print(f"Warning: Could not load state_dict from '{weights_path}': {e}")

    if not loaded:
        print("Warning: No trained probe weights found! Running with newly initialized weights.")

    model.eval()
    return model


def save_confusion_matrix_plots(class_grouped, overall_stats, output_dir="./confusion_matrices"):
    """
    Generates high-resolution color-coded confusion matrix plots for each source_class_label
    as well as the overall validation variant, and a combined 2x5 multi-panel summary grid image.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_classes = sorted(class_grouped.keys())

    # Items to plot: Each source class + Overall
    items = list(all_classes) + ['OVERALL']

    # 1. Combined Summary Grid Plot (2 rows x 5 cols)
    fig, axes = plt.subplots(2, 5, figsize=(26, 11))
    axes = axes.flatten()

    plt.suptitle("Color-Coded Confusion Matrices Per Source Class & Overall Variant", fontsize=20, fontweight='bold', y=0.98)

    for idx, item in enumerate(items):
        ax = axes[idx]
        if item == 'OVERALL':
            tn, fp, fn, tp = overall_stats['tn'], overall_stats['fp'], overall_stats['fn'], overall_stats['tp']
            title = f"OVERALL VALIDATION (N={overall_stats['total']})\nAcc: {overall_stats['acc']:.1f}% | AUC: {overall_stats['auc_str']}"
            single_filename = "cm_overall.png"
        else:
            c_data = class_grouped[item]
            c_name = DOTA_CLASS_NAMES.get(item, f"Class_{item}")
            t_labels = c_data["true_labels"]
            p_labels = c_data["pred_labels"]
            tn = sum(1 for y, p in zip(t_labels, p_labels) if y == 0 and p == 0)
            fp = sum(1 for y, p in zip(t_labels, p_labels) if y == 0 and p == 1)
            fn = sum(1 for y, p in zip(t_labels, p_labels) if y == 1 and p == 0)
            tp = sum(1 for y, p in zip(t_labels, p_labels) if y == 1 and p == 1)
            total = len(t_labels)
            acc = (tp + tn) / total * 100 if total > 0 else 0.0
            title = f"Class {item}: {c_name} (N={total})\nAcc: {acc:.1f}%"
            single_filename = f"cm_class_{item}_{c_name}.png"

        cm = np.array([[tn, fp], [fn, tp]])
        n_norm = tn + fp
        n_anom = fn + tp

        tn_pct = f"{tn/n_norm*100:.1f}%" if n_norm > 0 else "0%"
        fp_pct = f"{fp/n_norm*100:.1f}%" if n_norm > 0 else "0%"
        fn_pct = f"{fn/n_anom*100:.1f}%" if n_anom > 0 else "0%"
        tp_pct = f"{tp/n_anom*100:.1f}%" if n_anom > 0 else "0%"

        text_annotations = [
            [f"TN: {tn}\n({tn_pct})", f"FP: {fp}\n({fp_pct})"],
            [f"FN: {fn}\n({fn_pct})", f"TP: {tp}\n({tp_pct})"]
        ]

        # Draw Heatmap on Subplot
        sns.heatmap(cm, annot=False, fmt="d", cmap="YlGnBu", cbar=False, ax=ax, linewidths=2, linecolor='white')

        for r in range(2):
            for c in range(2):
                cell_color = "red" if (r != c and cm[r, c] > 0) else "darkgreen" if (r == c and cm[r, c] > 0) else "black"
                ax.text(c + 0.5, r + 0.5, text_annotations[r][c],
                        ha="center", va="center", color=cell_color, fontsize=12, fontweight='bold')

        ax.set_xticklabels(["Pred Normal (0)", "Pred Anom (1)"], fontsize=10, fontweight='bold')
        ax.set_yticklabels(["True Normal (0)", "True Anom (1)"], fontsize=10, fontweight='bold', rotation=0)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)

        # 2. Save Individual High-Res Confusion Matrix Plot
        fig_single, ax_single = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=False, fmt="d", cmap="YlGnBu", cbar=False, ax=ax_single, linewidths=3, linecolor='white')
        for r in range(2):
            for c in range(2):
                cell_color = "#b71c1c" if (r != c and cm[r, c] > 0) else "#1b5e20" if (r == c and cm[r, c] > 0) else "#212121"
                ax_single.text(c + 0.5, r + 0.5, text_annotations[r][c],
                               ha="center", va="center", color=cell_color, fontsize=16, fontweight='bold')

        ax_single.set_xticklabels(["Predicted Normal (0)", "Predicted Anomalous (1)"], fontsize=11, fontweight='bold')
        ax_single.set_yticklabels(["True Normal (0)", "True Anomalous (1)"], fontsize=11, fontweight='bold', rotation=0)
        ax_single.set_title(title, fontsize=14, fontweight='bold', pad=12)
        plt.tight_layout()
        single_path = os.path.join(output_dir, single_filename)
        plt.savefig(single_path, dpi=200, bbox_inches='tight')
        plt.close(fig_single)

    # Hide any unused subplots
    for idx in range(len(items), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    grid_path = os.path.join(output_dir, "cm_all_classes_grid.png")
    plt.savefig(grid_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined color-coded confusion matrix grid plot to '{grid_path}'")
    print(f"Saved individual color-coded confusion matrix plots to '{output_dir}/'")

    return grid_path


def generate_binary_stats(
    cache_path="./cached_features/val_block18_all_correct_unpooled_mc.pt",
    model_path="best_attention_probe.pt",
    weights_path="best_val_attention_weights.pt",
    output_path="binary_stats_report.md",
    plot_dir=".LinearProbe/confusion_matrices",
    batch_size=32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(cache_path):
        print(f"Error: Cache file '{cache_path}' does not exist!")
        return

    dataset = CachedFeatureDataset(cache_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    input_dim = dataset.features.shape[-1]
    model = load_probe_model(model_path, weights_path, input_dim, device)

    all_labels = []
    all_preds = []
    all_probs = []
    all_source_mc = []
    all_mc = []
    all_video_ids = []

    print("\nRunning forward pass through Attention Probe...")
    with torch.no_grad():
        for batch in dataloader:
            features = batch[0].to(device)
            labels = batch[1].to(device)
            mc_labels = batch[2]
            source_mc_labels = batch[3]
            vids = batch[4]

            logits, _ = model(features)
            probs = F.softmax(logits, dim=1)[:, 1]
            _, preds = logits.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_source_mc.extend(source_mc_labels.numpy())
            all_mc.extend(mc_labels.numpy())
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
    overall_tn = sum(1 for y, p in zip(all_labels, all_preds) if y == 0 and p == 0)
    overall_fp = sum(1 for y, p in zip(all_labels, all_preds) if y == 0 and p == 1)
    overall_fn = sum(1 for y, p in zip(all_labels, all_preds) if y == 1 and p == 0)
    overall_tp = sum(1 for y, p in zip(all_labels, all_preds) if y == 1 and p == 1)

    overall_acc = accuracy_score(all_labels, all_preds) * 100
    overall_prec = precision_score(all_labels, all_preds, zero_division=0) * 100
    overall_rec = recall_score(all_labels, all_preds, zero_division=0) * 100
    overall_f1 = f1_score(all_labels, all_preds, zero_division=0) * 100
    try:
        overall_auc = roc_auc_score(all_labels, all_probs)
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
    grid_plot_path = save_confusion_matrix_plots(class_grouped, overall_stats, output_dir=plot_dir)

    # Generate Markdown Report with Color-Coded 2x2 Matrix Cards
    report_lines = []
    report_lines.append("# Binary Attention Probe - Per Source Class Performance Report\n")
    report_lines.append(f"**Cached Features**: `{cache_path}`  ")
    report_lines.append(f"**Total Samples**: `{total_samples}`  \n")

    report_lines.append("## 1. Summary Table Per Source Class (Accident Category)\n")
    header_table = "| Class ID | Source Class Name | Total | Normal (0) | Anom (1) | TN | FP | FN | TP | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC |"
    divider_table = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
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
        n_normal = sum(1 for y in t_labels if y == 0)
        n_anom = sum(1 for y in t_labels if y == 1)

        tn = sum(1 for y, p in zip(t_labels, p_labels) if y == 0 and p == 0)
        fp = sum(1 for y, p in zip(t_labels, p_labels) if y == 0 and p == 1)
        fn = sum(1 for y, p in zip(t_labels, p_labels) if y == 1 and p == 0)
        tp = sum(1 for y, p in zip(t_labels, p_labels) if y == 1 and p == 1)

        acc = (tp + tn) / n_total * 100 if n_total > 0 else 0.0
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        try:
            auc = roc_auc_score(t_labels, c_probs)
            auc_str = f"{auc:.4f}"
        except Exception:
            auc_str = "N/A"

        row = f"| {src_id:<8} | {c_name:<25} | {n_total:<5} | {n_normal:<10} | {n_anom:<8} | {tn:<2} | {fp:<2} | {fn:<2} | {tp:<2} | {acc:<12.2f} | {prec:<13.2f} | {rec:<10.2f} | {f1:<12.2f} | {auc_str:<5} |"
        report_lines.append(row)
        print(row)

    overall_row = f"| **ALL**   | **OVERALL VALIDATION**    | {total_samples:<5} | {sum(1 for y in all_labels if y==0):<10} | {sum(1 for y in all_labels if y==1):<8} | {overall_tn:<2} | {overall_fp:<2} | {overall_fn:<2} | {overall_tp:<2} | {overall_acc:<12.2f} | {overall_prec:<13.2f} | {overall_rec:<10.2f} | {overall_f1:<12.2f} | {overall_auc_str:<5} |"
    report_lines.append(divider_table)
    report_lines.append(overall_row)
    print(divider_table)
    print(overall_row)

    report_lines.append("\n---\n")
    report_lines.append("## 2. Combined Color-Coded Confusion Matrix Grid\n")
    report_lines.append(f"![All Classes Confusion Matrix Grid]({plot_dir}/cm_all_classes_grid.png)\n\n---\n")

    report_lines.append("## 3. Color-Coded Confusion Matrices Per Source Class\n")

    # Insert individual 2x2 generated PNG plots
    for src_id in sorted(class_grouped.keys()):
        c_data = class_grouped[src_id]
        t_labels = c_data["true_labels"]
        p_labels = c_data["pred_labels"]
        c_name = DOTA_CLASS_NAMES.get(src_id, f"Class_{src_id}")

        n_total = len(t_labels)
        n_normal = sum(1 for y in t_labels if y == 0)
        n_anom = sum(1 for y in t_labels if y == 1)

        tn = sum(1 for y, p in zip(t_labels, p_labels) if y == 0 and p == 0)
        fp = sum(1 for y, p in zip(t_labels, p_labels) if y == 0 and p == 1)
        fn = sum(1 for y, p in zip(t_labels, p_labels) if y == 1 and p == 0)
        tp = sum(1 for y, p in zip(t_labels, p_labels) if y == 1 and p == 1)

        acc = (tp + tn) / n_total * 100 if n_total > 0 else 0.0
        single_img_path = f"{plot_dir}/cm_class_{src_id}_{c_name}.png"

        report_lines.append(f"### Class {src_id}: {c_name}")
        report_lines.append(f"**Total**: `{n_total}` | **Normal**: `{n_normal}` | **Anomalous**: `{n_anom}` | **Accuracy**: `{acc:.2f}%` | **TN**: `{tn}` | **FP**: `{fp}` | **FN**: `{fn}` | **TP**: `{tp}`  \n")
        report_lines.append(f"![Confusion Matrix Class {src_id} - {c_name}]({single_img_path})\n\n---\n")

    # Overall Confusion Matrix Section
    report_lines.append("## 4. Overall Color-Coded Confusion Matrix (Entire Variant)\n")
    report_lines.append(f"**Total Validation Samples**: `{total_samples}` | **Overall Accuracy**: `{overall_acc:.2f}%` | **Overall AUC**: `{overall_auc_str}` | **TN**: `{overall_tn}` | **FP**: `{overall_fp}` | **FN**: `{overall_fn}` | **TP**: `{overall_tp}`  \n")
    report_lines.append(f"![Overall Confusion Matrix]({plot_dir}/cm_overall.png)\n")

    with open(output_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nSaved detailed per-source-class performance report with color-coded matrices to '{output_path}'")

    # Generate Mean and Per-Head Attention Heatmaps overlaid on target frames
    heatmap_dir = "./attention_heatmaps_binary"
    try:
        from heatmaps import generate_attention_heatmaps_binary
    except ImportError:
        from LinearProbe.heatmaps import generate_attention_heatmaps_binary

    generate_attention_heatmaps_binary(
        weights_path=weights_path,
        sequence_dir="../DoTA_sequences",
        output_dir=heatmap_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate per source_class_label stats & color-coded confusion matrices for Attention Probe.")
    parser.add_argument("--cache_path", type=str, default="./cached_features/val_block18_all_correct_unpooled_mc.pt", help="Path to cached validation features")
    parser.add_argument("--model_path", type=str, default="best_attention_probe.pt", help="Path to saved Attention Probe state_dict")
    parser.add_argument("--weights_path", type=str, default="best_val_attention_weights.pt", help="Path to saved attention weights checkpoint")
    parser.add_argument("--output_path", type=str, default="binary_stats_report.md", help="Path to save output Markdown report")
    parser.add_argument("--plot_dir", type=str, default="./confusion_matrices", help="Directory to save confusion matrix plots")
    parser.add_argument("--heatmap_dir", type=str, default="./attention_heatmaps_binary", help="Directory to save attention heatmaps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")

    args = parser.parse_args()

    generate_binary_stats(
        cache_path=args.cache_path,
        model_path=args.model_path,
        weights_path=args.weights_path,
        output_path=args.output_path,
        plot_dir=args.plot_dir,
        batch_size=args.batch_size
    )
