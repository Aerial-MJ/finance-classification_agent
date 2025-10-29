import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
import os
import random
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
import numpy as np

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("main_text_image_k_fold.py requires the 'transformers' package. Install it via `pip install transformers`.") from exc

from model.resnet_bert import FineGrainedResNetTextFusion


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 42) -> None:
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-first; 若要严格复现可改为 deterministic 模式
    torch.backends.cudnn.benchmark = True


def load_text_mapping(json_path: Optional[str], data_root: str) -> Dict[str, str]:
    if not json_path:
        print("No text JSON provided; falling back to empty strings.")
        return {}
    json_path = os.path.expanduser(json_path)
    if not os.path.isfile(json_path):
        print(f"Text JSON not found at {json_path}; falling back to empty strings.")
        return {}
    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {json_path}, but received {type(data)}")

    def _normalize_key(key: str) -> str:
        return key.replace("\\", "/")

    normalized = {_normalize_key(k): str(v) for k, v in data.items()}
    print(f"Loaded {len(normalized)} text entries from {json_path}.")
    return normalized


class ImageTextFolder(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        text_mapping: Optional[Dict[str, str]],
        data_root: str,
        default_text: str = "",
        transform=None,
    ) -> None:
        super().__init__(root, transform=transform)
        self.data_root = os.path.abspath(data_root)
        self.text_mapping = text_mapping or {}
        self.default_text = default_text

        missing = 0
        for path, _ in self.samples:
            if self._lookup_text(path) is None:
                missing += 1
        if missing:
            print(f"Warning: {missing}/{len(self.samples)} images missing text; using default fallback.")

    def _lookup_text(self, path: str) -> Optional[str]:
        # Strategy 1: path relative to data_root
        rel_to_root = os.path.relpath(path, self.data_root).replace(os.sep, "/")
        if rel_to_root in self.text_mapping:
            return self.text_mapping[rel_to_root]

        # Strategy 2: path relative to self.root
        rel_local = os.path.relpath(path, self.root).replace(os.sep, "/")
        if rel_local in self.text_mapping:
            return self.text_mapping[rel_local]

        # Strategy 3: try "raw/category/filename"
        parts = rel_to_root.split("/")
        if len(parts) >= 2:
            category_and_file = "/".join(parts[-2:])
            alternative_key = f"raw/{category_and_file}"
            if alternative_key in self.text_mapping:
                return self.text_mapping[alternative_key]

        return None

    def __getitem__(self, index):  # type: ignore[override]
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        text = self._lookup_text(path)
        if text is None:
            text = self.default_text
        return sample, text, target


def make_collate_fn(tokenizer: AutoTokenizer, max_length: int):
    def _collate(batch: Iterable):
        images, texts, labels = zip(*batch)
        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        tokenized = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "images": images_tensor,
            "text": tokenized,
            "labels": labels_tensor,
        }

    return _collate


def create_fold_dataloaders(
    train_dataset: ImageTextFolder,
    val_dataset: ImageTextFolder,
    train_indices: List[int],
    val_indices: List[int],
    tokenizer: AutoTokenizer,
    batch_size: int,
    num_workers: int,
    max_text_length: int,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for a single fold."""
    collate = make_collate_fn(tokenizer, max_text_length)

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,            # 更稳的 BN
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    return train_loader, val_loader


def build_dataset(
    data_root: str,
    text_mapping: Dict[str, str],
    img_size: int,
    default_text: str,
    split: str = "raw",
    is_training: bool = True,
):
    """Build dataset for k-fold cross-validation."""
    data_dir = os.path.join(data_root, split)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Expected {split}/ directory under {data_root}")

    if is_training:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.15)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    dataset = ImageTextFolder(
        data_dir,
        text_mapping=text_mapping,
        data_root=data_root,
        default_text=default_text,
        transform=transform,
    )

    return dataset


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        text_inputs = {k: v.to(device, non_blocking=True) for k, v in batch["text"].items()}
        text_kwargs = {"input_ids": text_inputs["input_ids"]}
        if "attention_mask" in text_inputs:
            text_kwargs["attention_mask"] = text_inputs["attention_mask"]
        if "token_type_ids" in text_inputs:
            text_kwargs["token_type_ids"] = text_inputs["token_type_ids"]

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(images, **text_kwargs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images, **text_kwargs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        text_inputs = {k: v.to(device, non_blocking=True) for k, v in batch["text"].items()}
        text_kwargs = {"input_ids": text_inputs["input_ids"]}
        if "attention_mask" in text_inputs:
            text_kwargs["attention_mask"] = text_inputs["attention_mask"]
        if "token_type_ids" in text_inputs:
            text_kwargs["token_type_ids"] = text_inputs["token_type_ids"]

        logits = model(images, **text_kwargs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for target, pred in zip(labels.view(-1), preds.view(-1)):
            cm[target.long(), pred.long()] += 1

    accuracy = correct / total if total else 0.0
    return total_loss / total if total else 0.0, accuracy, cm


def per_class_accuracy(cm: torch.Tensor) -> torch.Tensor:
    return cm.diag().float() / cm.sum(dim=1).clamp_min(1).float()


def save_confusion_fig(
    matrix: torch.Tensor,
    class_names: Sequence[str],
    out_path: str,
    normalized: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("matplotlib is required to save the confusion matrix figure.") from exc

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    data = matrix.cpu().numpy()
    fig_size = max(6.0, len(class_names) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    cmap = "Blues"
    im = ax.imshow(data, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    title = "Row-normalized Confusion Matrix" if normalized else "Confusion Matrix"
    ax.set_title(title)
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Predicted")

    thresh = data.max() / 2.0 if data.size else 0.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = data[i, j]
            text = f"{value:.2f}" if normalized else f"{int(round(value))}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K-Fold Cross-Validation (train:val = 1:(k-1)) for Multimodal Classification"
    )
    parser.add_argument("--data_root", type=str, default="data_raw", help="Root directory containing raw/ folder with image categories.")
    parser.add_argument("--text_json", type=str, default="data_raw/text_raw.json", help="JSON file with image-relative paths mapped to text.")
    parser.add_argument("--missing_text_fallback", type=str, default="", help="Fallback text when no entry is found.")
    parser.add_argument("--text_model_name", type=str, default="./bert-base-chinese", help="Hugging Face model name for the text encoder.")
    parser.add_argument("--train_text_encoder", action="store_true", help="Fine-tune the text encoder instead of freezing it.")
    parser.add_argument("--fusion_dim", type=int, default=768, help="Projection dimension for each modality before fusion.")
    parser.add_argument("--max_text_length", type=int, default=512, help="Maximum number of tokens for the tokenizer.")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_image", action="store_true", default=True, help="Use ImageNet weights for the image encoder.")
    parser.add_argument("--freeze_image_encoder", action="store_true", default=True, help="Freeze the image encoder backbone.")
    parser.add_argument("--save_dir", type=str, default="kfold_checkpoints", help="Directory to store fold checkpoints.")
    parser.add_argument("--normalize_confusion", action="store_true", help="Print row-normalized confusion matrix.")
    parser.add_argument("--save_confusion_fig", action="store_true", help="Save confusion matrix figures for each fold.")
    parser.add_argument("--check_data_leakage", action="store_true", help="Enable data leakage checks based on resolved paths.")
    return parser.parse_args()


def train_single_fold(
    fold: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[float, torch.Tensor]:
    """Train a single fold and return best validation accuracy and confusion matrix."""
    print(f"\n{'='*80}")
    print(f"Training Fold {fold + 1}/{args.k_folds}  (scheme: train=1 fold, val={args.k_folds-1} folds)")
    print(f"{'='*80}")

    model = FineGrainedResNetTextFusion(
        num_classes=num_classes,
        pretrained_image=args.pretrained_image,
        dropout=args.dropout,
        text_model_name=args.text_model_name,
        text_trainable=args.train_text_encoder,
        fusion_dim=args.fusion_dim,
        max_text_length=args.max_text_length,
    ).to(device)

    # Freeze image/text encoders if requested; set eval() to lock BN/Dropout
    if args.freeze_image_encoder:
        model.image_encoder.eval()
        for p in model.image_encoder.parameters():
            p.requires_grad = False
        for p in model.image_proj.parameters():
            p.requires_grad = True
    else:
        for p in model.image_encoder.parameters():
            p.requires_grad = True

    if not args.train_text_encoder and hasattr(model, "text_encoder"):
        model.text_encoder.eval()
        for p in model.text_encoder.parameters():
            p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    best_acc = -1.0
    best_cm: Optional[torch.Tensor] = None
    save_path = os.path.join(args.save_dir, f"fold_{fold + 1}_best.pt")
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc, cm = evaluate(model, val_loader, criterion, device, num_classes)
        scheduler.step()

        print(
            f"  [{epoch:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.6f} | time={time.time() - t0:.1f}s"
        )

        if best_cm is None or val_acc > best_acc:
            best_acc = val_acc
            best_cm = cm.detach().cpu()
            torch.save(
                {
                    "fold": fold + 1,
                    "state_dict": model.state_dict(),
                    "acc": float(best_acc),
                    "epoch": epoch,
                    "args": vars(args),
                },
                save_path,
            )
            print(f"    -> Saved best checkpoint for fold {fold + 1} (acc={best_acc:.4f})")

    if best_cm is None:
        # Fallback (should not happen)
        _, _, cm = evaluate(model, val_loader, criterion, device, num_classes)
        best_cm = cm.detach().cpu()
        best_acc = (best_cm.diag().sum().item()) / best_cm.sum().item()

    print(f"\nFold {fold + 1} Best Validation Accuracy: {best_acc:.4f}")
    return best_acc, best_cm


def main() -> None:
    args = parse_args()
    set_seed(42)

    # Default text_json if needed
    if args.text_json is None:
        args.text_json = os.path.join(args.data_root, "text_raw.json")
        print(f"Using default text_json path: {args.text_json}")

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    text_mapping = load_text_mapping(args.text_json, args.data_root)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Base dataset (training transform) — only to read classes and the master samples list
    base_dataset = build_dataset(
        args.data_root,
        text_mapping,
        args.img_size,
        args.missing_text_fallback,
        split="raw",
        is_training=True,
    )

    num_classes = len(base_dataset.classes)
    idx_to_class = {i: c for i, c in enumerate(base_dataset.classes)}

    print("Class mapping:")
    for idx in range(len(idx_to_class)):
        print(f"  {idx}: {idx_to_class[idx]}")

    print(f"\nTotal samples in raw/: {len(base_dataset)}")
    print(f"K-Fold (train:val = 1:{args.k_folds-1}) with k={args.k_folds}")

    # Prepare folds with stratification
    labels = np.array([label for _, label in base_dataset.samples])
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    # Collect fold index arrays first
    fold_indices: List[np.ndarray] = []
    for _, val_idx in skf.split(np.zeros(len(labels)), labels):
        fold_indices.append(val_idx)

    # Results
    fold_accuracies: List[float] = []
    fold_confusion_matrices: List[torch.Tensor] = []

    for fold in range(args.k_folds):
        # Core change: TRAIN uses only fold i; VAL uses all other folds
        train_indices = fold_indices[fold]
        val_indices = np.concatenate([fold_indices[j] for j in range(args.k_folds) if j != fold], axis=0)

        print(f"\nFold {fold + 1}: Train={len(train_indices)}, Val={len(val_indices)}")

        # Build actual datasets for this fold
        train_dataset = build_dataset(
            args.data_root,
            text_mapping,
            args.img_size,
            args.missing_text_fallback,
            split="raw",
            is_training=True,
        )
        val_dataset = build_dataset(
            args.data_root,
            text_mapping,
            args.img_size,
            args.missing_text_fallback,
            split="raw",
            is_training=False,
        )

        # Robust index remapping by PATH to avoid any order mismatch
        base_paths = [p for p, _ in base_dataset.samples]
        p2i_train = {p: i for i, (p, _) in enumerate(train_dataset.samples)}
        p2i_val = {p: i for i, (p, _) in enumerate(val_dataset.samples)}

        try:
            train_indices = np.array([p2i_train[base_paths[i]] for i in train_indices], dtype=np.int64)
            val_indices = np.array([p2i_val[base_paths[i]] for i in val_indices], dtype=np.int64)
        except KeyError as e:
            raise RuntimeError(f"Index remapping failed due to path mismatch: {e}")

        # Optional leakage check based on final resolved paths
        if args.check_data_leakage:
            train_paths = {train_dataset.samples[i][0] for i in train_indices.tolist()}
            val_paths = {val_dataset.samples[i][0] for i in val_indices.tolist()}
            overlap = train_paths & val_paths
            if overlap:
                print(f"   [DEBUG] overlap after remap: {len(overlap)} (should be 0)")
                for k, p in enumerate(list(overlap)[:10], 1):
                    print(f"     {k}. {p}")
            else:
                print(f"   [DEBUG] No overlap after remap (train={len(train_paths)}, val={len(val_paths)})")

        # Dataloaders
        train_loader, val_loader = create_fold_dataloaders(
            train_dataset,
            val_dataset,
            train_indices.tolist(),
            val_indices.tolist(),
            tokenizer,
            args.batch_size,
            args.num_workers,
            args.max_text_length,
        )

        # Train one fold
        best_acc, best_cm = train_single_fold(
            fold, train_loader, val_loader, num_classes, args, device
        )

        fold_accuracies.append(best_acc)
        fold_confusion_matrices.append(best_cm)

        # Save confusion fig per fold (optional)
        if args.save_confusion_fig:
            class_names = [idx_to_class[idx] for idx in range(num_classes)]
            cm_fig_path = os.path.join(args.save_dir, f"fold_{fold + 1}_confusion_matrix.png")

            if args.normalize_confusion:
                row_sums = best_cm.sum(dim=1, keepdim=True).clamp_min(1)
                normalized_cm = best_cm.float() / row_sums
                save_confusion_fig(normalized_cm, class_names, cm_fig_path, normalized=True)
            else:
                save_confusion_fig(best_cm.float(), class_names, cm_fig_path, normalized=False)

            print(f"Saved confusion matrix figure to {cm_fig_path}")

    # Summary
    print("\n" + "=" * 80)
    print("K-Fold Cross-Validation Results (train:val = 1:(k-1))")
    print("=" * 80)

    for fold, acc in enumerate(fold_accuracies):
        print(f"Fold {fold + 1}: {acc:.4f}")

    mean_acc = float(np.mean(fold_accuracies)) if len(fold_accuracies) else 0.0
    std_acc = float(np.std(fold_accuracies)) if len(fold_accuracies) else 0.0
    print(f"\nMean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Min Accuracy:  {float(np.min(fold_accuracies)):.4f}")
    print(f"Max Accuracy:  {float(np.max(fold_accuracies)):.4f}")

    # Aggregate confusion matrix (sum across folds)
    aggregated_cm = None
    for cm in fold_confusion_matrices:
        if aggregated_cm is None:
            aggregated_cm = cm.clone()
        else:
            aggregated_cm += cm

    if aggregated_cm is None:
        aggregated_cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

    print("\nAggregated Confusion Matrix (sum of all folds):")
    print(aggregated_cm)

    # Per-class accuracy from aggregated confusion matrix
    class_acc = per_class_accuracy(aggregated_cm).cpu().numpy()
    print("\nPer-class accuracy (aggregated):")
    for idx, acc in enumerate(class_acc):
        print(f"{idx:02d} ({idx_to_class[idx]}): {acc:.4f}")

    # Save aggregated confusion matrix figure (optional)
    if args.save_confusion_fig:
        class_names = [idx_to_class[idx] for idx in range(num_classes)]
        agg_cm_path = os.path.join(args.save_dir, "aggregated_confusion_matrix.png")

        if args.normalize_confusion:
            row_sums = aggregated_cm.sum(dim=1, keepdim=True).clamp_min(1)
            normalized_agg_cm = aggregated_cm.float() / row_sums
            save_confusion_fig(normalized_agg_cm, class_names, agg_cm_path, normalized=True)
        else:
            save_confusion_fig(aggregated_cm.float(), class_names, agg_cm_path, normalized=False)

        print(f"\nSaved aggregated confusion matrix to {agg_cm_path}")

    # Save summary JSON
    os.makedirs(args.save_dir, exist_ok=True)
    summary_path = os.path.join(args.save_dir, "kfold_summary.json")
    summary = {
        "k_folds": args.k_folds,
        "scheme": "train:val = 1:(k-1)",
        "fold_accuracies": [float(acc) for acc in fold_accuracies],
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "min_accuracy": float(np.min(fold_accuracies)) if len(fold_accuracies) else 0.0,
        "max_accuracy": float(np.max(fold_accuracies)) if len(fold_accuracies) else 0.0,
        "per_class_accuracy": {idx_to_class[idx]: float(acc) for idx, acc in enumerate(class_acc)},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved k-fold summary to {summary_path}")


if __name__ == "__main__":
    main()
