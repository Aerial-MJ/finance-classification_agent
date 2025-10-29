import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from model.resnet_bert import FineGrainedResNetTextFusion


def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple:
    """Load checkpoint and extract model configuration."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "args" not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain 'args' field")

    args = checkpoint["args"]
    state_dict = checkpoint["state_dict"]

    print(f"Loaded checkpoint from fold {checkpoint.get('fold', 'N/A')}")
    print(f"Best validation accuracy: {checkpoint.get('acc', 'N/A'):.4f}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")

    return state_dict, args


def build_model(args: Dict, num_classes: int, device: torch.device) -> FineGrainedResNetTextFusion:
    """Build model from configuration."""
    model = FineGrainedResNetTextFusion(
        num_classes=num_classes,
        pretrained_image=args.get("pretrained_image", True),
        dropout=args.get("dropout", 0.3),
        text_model_name=args.get("text_model_name", "./bert-base-chinese"),
        text_trainable=args.get("train_text_encoder", False),
        fusion_dim=args.get("fusion_dim", 768),
        max_text_length=args.get("max_text_length", 512),
    )
    return model.to(device)


def get_image_transform(img_size: int = 224) -> transforms.Compose:
    """Get image preprocessing transform."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_class_names(data_root: str) -> List[str]:
    """Load class names from data directory."""
    raw_dir = os.path.join(data_root, "raw")
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Expected raw/ directory under {data_root}")

    class_names = sorted([d for d in os.listdir(raw_dir)
                         if os.path.isdir(os.path.join(raw_dir, d))])
    return class_names


def load_text_mapping(json_path: str) -> Dict[str, str]:
    """Load text mapping from JSON file."""
    if not os.path.isfile(json_path):
        print(f"Warning: Text mapping file not found at {json_path}")
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        text_mapping = json.load(f)

    print(f"Loaded {len(text_mapping)} text entries from {json_path}")
    return text_mapping


def lookup_text_for_image(
    image_path: str,
    text_mapping: Dict[str, str],
    data_root: str,
    default_text: str = ""
) -> str:
    """Lookup text for an image path using multiple strategies."""
    data_root = os.path.abspath(data_root)
    image_path_abs = os.path.abspath(image_path)

    # Normalize key function
    def normalize_key(key: str) -> str:
        return key.replace("\\", "/")

    # Strategy 1: path relative to data_root
    try:
        rel_to_root = os.path.relpath(image_path_abs, data_root).replace(os.sep, "/")
        rel_to_root_norm = normalize_key(rel_to_root)
        if rel_to_root_norm in text_mapping:
            return text_mapping[rel_to_root_norm]
    except ValueError:
        pass

    # Strategy 2: try "raw/category/filename" pattern
    parts = Path(image_path_abs).parts
    if len(parts) >= 2:
        category_and_file = f"{parts[-2]}/{parts[-1]}"
        alternative_key = f"raw/{category_and_file}"
        alternative_key_norm = normalize_key(alternative_key)
        if alternative_key_norm in text_mapping:
            return text_mapping[alternative_key_norm]

    # Strategy 3: try filename only
    filename = os.path.basename(image_path)
    for key, value in text_mapping.items():
        if key.endswith(filename):
            return value

    print(f"Warning: No text found for {image_path}, using default: '{default_text}'")
    return default_text


@torch.no_grad()
def predict_single_sample(
    model: FineGrainedResNetTextFusion,
    image_path: str,
    text: str,
    tokenizer: AutoTokenizer,
    transform: transforms.Compose,
    device: torch.device,
    max_text_length: int = 512,
) -> tuple:
    """Predict on a single image-text pair."""
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Tokenize text
    text_inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt"
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    # Prepare text kwargs
    text_kwargs = {"input_ids": text_inputs["input_ids"]}
    if "attention_mask" in text_inputs:
        text_kwargs["attention_mask"] = text_inputs["attention_mask"]
    if "token_type_ids" in text_inputs:
        text_kwargs["token_type_ids"] = text_inputs["token_type_ids"]

    # Forward pass
    logits = model(image_tensor, **text_kwargs)
    probs = F.softmax(logits, dim=1)

    pred_idx = logits.argmax(dim=1).item()
    confidence = probs[0, pred_idx].item()

    return pred_idx, confidence, probs[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Single-sample inference for multimodal classification model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g., kfold_checkpoints/fold_1_best.pt)"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Input text (if not provided, will lookup from text_mapping)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data_raw",
        help="Root directory containing raw/ folder (for class names and text lookup)"
    )
    parser.add_argument(
        "--text_json",
        type=str,
        default=None,
        help="JSON file with text mapping (default: data_root/text_raw.json)"
    )
    parser.add_argument(
        "--missing_text_fallback",
        type=str,
        default="",
        help="Fallback text when no entry is found"
    )
    parser.add_argument(
        "--show_top_k",
        type=int,
        default=3,
        help="Show top-k predictions"
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    state_dict, train_args = load_checkpoint(args.checkpoint, device)

    # Load class names
    print(f"\nLoading class names from {args.data_root}...")
    class_names = load_class_names(args.data_root)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")

    # Build model
    print("\nBuilding model...")
    model = build_model(train_args, num_classes, device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully")

    # Load tokenizer
    text_model_name = train_args.get("text_model_name", "./bert-base-chinese")
    print(f"\nLoading tokenizer from {text_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    # Get image transform
    img_size = train_args.get("img_size", 224)
    transform = get_image_transform(img_size)

    # Determine text
    if args.text is not None:
        text = args.text
        print(f"\nUsing provided text: {text}")
    else:
        # Load text mapping and lookup
        text_json = args.text_json or os.path.join(args.data_root, "text_raw.json")
        print(f"\nLoading text mapping from {text_json}...")
        text_mapping = load_text_mapping(text_json)

        text = lookup_text_for_image(
            args.image,
            text_mapping,
            args.data_root,
            args.missing_text_fallback
        )
        print(f"Looked up text: {text}")

    # Predict
    print(f"\nRunning inference on {args.image}...")
    pred_idx, confidence, all_probs = predict_single_sample(
        model,
        args.image,
        text,
        tokenizer,
        transform,
        device,
        max_text_length=train_args.get("max_text_length", 512)
    )

    # Display results
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    print(f"Image: {args.image}")
    print(f"Text: {text}")
    print(f"\nPredicted Class: {class_names[pred_idx]} (index: {pred_idx})")
    print(f"Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")

    # Show top-k predictions
    print(f"\nTop-{args.show_top_k} Predictions:")
    top_k_indices = all_probs.argsort()[::-1][:args.show_top_k]
    for rank, idx in enumerate(top_k_indices, 1):
        prob = all_probs[idx]
        print(f"  {rank}. {class_names[idx]:<30} {prob:.4f} ({prob * 100:.2f}%)")

    print("=" * 80)


if __name__ == "__main__":
    main()
