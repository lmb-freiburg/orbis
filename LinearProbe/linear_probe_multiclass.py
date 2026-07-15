import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from LinearProbe.dota import get_dota_dataloaders
from util import instantiate_from_config


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=18):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


def load_model_from_config(exp_dir, config_path, ckpt_path, device):
    config_file = (Path(exp_dir) / config_path).resolve()
    ckpt_file = (Path(exp_dir) / ckpt_path).resolve()

    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    cfg = OmegaConf.load(config_file)
    model = instantiate_from_config(cfg.model)
    state_dict = torch.load(str(ckpt_file), map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def collect_multiclass_subset(loader):
    base_dataset = loader.dataset.dataset
    subset_indices = [
        index
        for index in loader.dataset.indices
        if base_dataset.samples[index].get("class_label", None) is not None
        and int(base_dataset.samples[index]["class_label"]) >= 0
    ]
    return torch.utils.data.Subset(base_dataset, subset_indices)


def pool_activation(features):
    if features.dim() == 4:
        return features.amax(dim=(1, 2))
    if features.dim() == 3:
        return features.amax(dim=1)
    return features


def infer_features(model, clips, device, activation, hook_name):
    clips = clips.to(device)

    if hasattr(model, "encode_frames") and hasattr(model, "vit"):
        clips = clips.permute(0, 2, 1, 3, 4).contiguous()
        latents = model.encode_frames(clips)
        context = latents[:, :-1].contiguous() if latents.size(1) > 1 else None
        target = latents[:, -1:].contiguous()

        timestep = torch.zeros(clips.shape[0], dtype=torch.float32, device=device)
        frame_rate = torch.full((clips.shape[0],), 5.0, dtype=torch.float32, device=device)
        _ = model.vit(target, context, timestep, frame_rate=frame_rate)
    else:
        timestep = torch.zeros(clips.shape[0], dtype=torch.float32, device=device)
        _ = model(clips, timestep)

    if hook_name not in activation:
        raise RuntimeError("Feature hook did not capture an activation tensor.")

    return pool_activation(activation[hook_name])


def evaluate(model, probe, dataloader, device, activation, hook_name, num_classes):
    probe.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for clips, _, class_labels in dataloader:
            features = infer_features(model, clips, device, activation, hook_name)
            logits = probe(features)
            predictions = logits.argmax(dim=1)

            all_labels.extend(class_labels.cpu().tolist())
            all_preds.extend(predictions.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    return {
        "accuracy": accuracy,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "confusion_matrix": matrix,
    }


def train_linear_probe(
    exp_dir="./logs_wm/orbis_288x512",
    config_path="config.yaml",
    ckpt_path="checkpoints/last.ckpt",
    seq_dir="../DoTA_sequences",
    anno_dir="../DOTA_annotations",
    num_classes=18,
    block_index=17,
    epochs=50,
    lr=1e-3,
    weight_decay=0.01,
    batch_size=64,
    num_workers=4,
    max_samples=900,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dota_dataloaders(
        seq_dir,
        anno_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
        return_multiclass_labels=True,
    )

    train_subset = collect_multiclass_subset(train_loader)
    val_subset = collect_multiclass_subset(val_loader)

    if len(train_subset) == 0 or len(val_subset) == 0:
        raise ValueError("No anomaly clips with valid class labels were found for multiclass training.")

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = load_model_from_config(exp_dir, config_path, ckpt_path, device)
    backbone = getattr(model, "vit", model)
    if not hasattr(backbone, "blocks"):
        raise AttributeError("Expected the loaded model to expose a `blocks` module list.")

    activation = {}

    def get_activation(name):
        def hook(_module, _input, output):
            activation[name] = output[0] if isinstance(output, tuple) else output

        return hook

    hook_name = f"block_{block_index + 1}"
    hook_handle = backbone.blocks[block_index].register_forward_hook(get_activation(hook_name))

    probe = nn.LazyLinear(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    try:
        for epoch in range(epochs):
            probe.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for clips, _, class_labels in train_loader:
                class_labels = class_labels.to(device)

                optimizer.zero_grad()
                features = infer_features(model, clips, device, activation, hook_name)
                logits = probe(features)
                loss = criterion(logits, class_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predictions = logits.argmax(dim=1)
                total += class_labels.size(0)
                correct += predictions.eq(class_labels).sum().item()

            train_acc = 100.0 * correct / max(total, 1)
            val_metrics = evaluate(model, probe, val_loader, device, activation, hook_name, num_classes)

            print(
                f"\nEpoch {epoch + 1}/{epochs} | Train Loss: {total_loss / max(len(train_loader), 1):.4f} | "
                f"Train Acc: {train_acc:.2f}%"
            )
            print(
                f"--> Val Acc: {val_metrics['accuracy']:.2f}% | Precision: {val_metrics['precision']:.2f}% | "
                f"Recall: {val_metrics['recall']:.2f}% | F1: {val_metrics['f1']:.2f}%"
            )
            print(f"--> Confusion Matrix:\n{val_metrics['confusion_matrix']}")
    finally:
        hook_handle.remove()


def parse_args():
    parser = argparse.ArgumentParser(description="Train an 18-class linear probe for DOTA anomaly labels.")
    parser.add_argument("--exp_dir", type=str, default="./logs_wm/orbis_288x512")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.ckpt")
    parser.add_argument("--seq_dir", type=str, default="../DoTA_sequences")
    parser.add_argument("--anno_dir", type=str, default="../DOTA_annotations")
    parser.add_argument("--num_classes", type=int, default=18)
    parser.add_argument("--block_index", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=900)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_linear_probe(
        exp_dir=args.exp_dir,
        config_path=args.config,
        ckpt_path=args.ckpt,
        seq_dir=args.seq_dir,
        anno_dir=args.anno_dir,
        num_classes=args.num_classes,
        block_index=args.block_index,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )


class LinearProbe(nn.Module):
	def __init__(self, input_dim, num_classes=18):
		super().__init__()
		self.classifier = nn.Linear(input_dim, num_classes)

	def forward(self, x):
		return self.classifier(x)


def remap_labels(dataset, label_mapping):
	remapped = torch.tensor([label_mapping[int(label)] for label in dataset.labels.tolist()], dtype=torch.long)
	dataset.labels = remapped


def build_label_mapping(train_dataset, val_dataset, num_classes):
	raw_labels = torch.cat([train_dataset.labels, val_dataset.labels], dim=0)
	valid_labels = sorted({int(label) for label in raw_labels.tolist() if int(label) >= 0})

	if not valid_labels:
		raise ValueError('No valid class labels were found in the cached features.')

	if valid_labels == list(range(len(valid_labels))):
		label_mapping = {label: label for label in valid_labels}
	elif valid_labels == list(range(1, len(valid_labels) + 1)):
		label_mapping = {label: label - 1 for label in valid_labels}
	else:
		label_mapping = {label: idx for idx, label in enumerate(valid_labels)}

	if len(label_mapping) > num_classes:
		raise ValueError(f'Found {len(label_mapping)} classes, but the classifier was configured for only {num_classes}.')

	return label_mapping


def evaluate(model, dataloader, device):
	model.eval()
	all_labels = []
	all_preds = []

	with torch.no_grad():
		for features, labels in dataloader:
			features = features.to(device)
			labels = labels.to(device)

			outputs = model(features)
			predictions = outputs.argmax(dim=1)

			all_labels.extend(labels.cpu().tolist())
			all_preds.extend(predictions.cpu().tolist())

	accuracy = accuracy_score(all_labels, all_preds) * 100
	precision, recall, f1, _ = precision_recall_fscore_support(
		all_labels,
		all_preds,
		average='macro',
		zero_division=0,
	)
	matrix = confusion_matrix(all_labels, all_preds)

	return {
		'accuracy': accuracy,
		'precision': precision * 100,
		'recall': recall * 100,
		'f1': f1 * 100,
		'confusion_matrix': matrix,
	}

