import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set directory structure
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DIAGNOSTIC_PROBES_DIR = os.path.dirname(SCRIPTS_DIR)
PROJECT_ROOT = os.path.dirname(DIAGNOSTIC_PROBES_DIR)

OUTPUT_DIR = os.path.join(DIAGNOSTIC_PROBES_DIR, "comparisonPlots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_markdown_table(filepath):
    if not os.path.isabs(filepath):
        if os.path.exists(os.path.join(DIAGNOSTIC_PROBES_DIR, filepath)):
            filepath = os.path.join(DIAGNOSTIC_PROBES_DIR, filepath)
        elif os.path.exists(os.path.join(PROJECT_ROOT, filepath)):
            filepath = os.path.join(PROJECT_ROOT, filepath)
        else:
            filepath = os.path.join(DIAGNOSTIC_PROBES_DIR, filepath)
        
    acc_list = []
    auc_list = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    in_table = False
    for line in lines:
        if line.startswith('| Class ID |'):
            in_table = True
            continue
        if in_table and line.startswith('|---'):
            continue
        
        if in_table and line.startswith('|'):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 13:
                class_id_str = parts[1]
                if class_id_str.isdigit() or '**ALL**' in class_id_str:
                    acc = round(float(parts[8].replace('%', '')) / 100.0, 2)
                    auc = round(float(parts[12]), 2)
                    acc_list.append(acc)
                    auc_list.append(auc)
                    
                    if '**ALL**' in class_id_str:
                        break # Stop after overall
    return acc_list, auc_list

binary_acc, binary_auc = parse_markdown_table(os.path.join("reports", "stats_report_binary.md"))
surprise_acc, surprise_auc = parse_markdown_table(os.path.join("reports", "stats_report_surprise_combined_t3.md"))

vjepa_acc = [63.64, 70.48, 75.0, 69.23, 67.08, 78.57, 80.0, 63.16, 55.32, 68.17]
vjepa_acc = [round(x / 100.0, 2) for x in vjepa_acc]
vjepa_auc = [0.6333, 0.7959, 0.8169, 0.7867, 0.7603, 0.8125, 0.7917, 0.6583, 0.6935, 0.7580]
vjepa_auc = [round(x, 2) for x in vjepa_auc]

# Data extracted from the markdown reports
classes = [
    "start_stop_or_\nstationary\n(n=11)",
    "moving_ahead_\nor_waiting\n(n=105)",
    "lateral\n(n=80)",
    "oncoming\n(n=52)",
    "turning\n(n=243)",
    "pedestrian\n(n=14)",
    "obstacle\n(n=10)",
    "leave_to_right\n(n=38)",
    "leave_to_left\n(n=47)",
    "OVERALL\n(n=600)"
]

data_acc = {
    'Orbis Activations': binary_acc,
    'VJEPA 2.1': vjepa_acc,
    'Orbis Suprise Scores': surprise_acc
}

data_auc = {
    'Orbis Activations': binary_auc,
    'VJEPA 2.1': vjepa_auc,
    'Orbis Suprise Scores': surprise_auc
}

df_acc = pd.DataFrame(data_acc, index=classes)
df_auc = pd.DataFrame(data_auc, index=classes)

fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

sns.heatmap(df_acc, annot=True, fmt=".2f", cmap="viridis", cbar=False, ax=axes[0], annot_kws={"weight": "bold", "size": 14})
axes[0].set_title("Per Source Class Accuracy Comparison", fontsize=18, fontweight='bold', pad=15)
axes[0].set_ylabel("Source Class", fontsize=16, fontweight='bold', labelpad=10)
axes[0].set_xlabel("Model Variant", fontsize=16, fontweight='bold', labelpad=10)
axes[0].set_yticklabels(axes[0].get_yticklabels(), fontweight='bold', fontsize=14)
axes[0].set_xticklabels(axes[0].get_xticklabels(), fontweight='bold', fontsize=14)

sns.heatmap(df_auc, annot=True, fmt=".2f", cmap="viridis", cbar=False, ax=axes[1], annot_kws={"weight": "bold", "size": 14})
axes[1].set_title("Per Source Class AUC Comparison", fontsize=18, fontweight='bold', pad=15)
axes[1].set_ylabel("")
axes[1].set_xlabel("Model Variant", fontsize=16, fontweight='bold', labelpad=10)
axes[1].set_xticklabels(axes[1].get_xticklabels(), fontweight='bold', fontsize=14)

plt.tight_layout(w_pad=3.0)
heatmap_save_path = os.path.join(OUTPUT_DIR, "combined_comparison_heatmap.png")
plt.savefig(heatmap_save_path)
print(f"Saved {heatmap_save_path}")

dataset_counts = {
    'start_stop_or_stationary': 68,
    'moving_ahead_or_waiting': 452,
    'lateral': 475,
    'oncoming': 294,
    'turning': 1121,
    'pedestrian': 65,
    'obstacle': 45,
    'leave_to_right': 249,
    'leave_to_left': 231
}

plt.figure(figsize=(18, 8))
keys = [k.replace('_', '\n') for k in dataset_counts.keys()]
bars = plt.bar(keys, dataset_counts.values(), color='#2980b9', edgecolor='black', width=0.85)
plt.title("Overall Dataset Distribution by Source Class (N=3000)", fontsize=22, fontweight='bold', pad=20)
plt.xticks(rotation=0, ha='center', fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 15, int(yval), ha='center', va='bottom', fontsize=16, fontweight='bold')

plt.tight_layout()
dist_bar_save_path = os.path.join(OUTPUT_DIR, "dataset_class_distribution_bar.png")
plt.savefig(dist_bar_save_path, dpi=150)
print(f"Saved {dist_bar_save_path}")
