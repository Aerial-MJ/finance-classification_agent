import warnings
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from cnocr import CnOcr
from torchvision import models
from torchvision.transforms.functional import to_pil_image


class ResNetOCRHybrid(nn.Module):

    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = True,
        dropout: float = 0.3,
        ocr_model_path: Optional[str] = None,
        ocr_model_name: Optional[str] = "densenet_lite_136-gru",
        ocr_detector_name: Optional[str] = "db_resnet18",
        ocr_context: str = "cuda",
        ocr_max_tokens: int = 64,
        ocr_embedding_dim: int = 256,
        freeze_vision_backbone: bool = False,
        freeze_resnet_backbone: bool = True,
        image_mean: Optional[Iterable[float]] = None,
        image_std: Optional[Iterable[float]] = None,
    ) -> None:
        super().__init__()

        if ocr_model_path:
            warnings.warn(
                "Parameter 'ocr_model_path' is deprecated. "
                "Please use 'ocr_model_name' to select the lightweight OCR model.",
                stacklevel=2,
            )

        resnet_weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=resnet_weights)
        self.resnet_conv = nn.Sequential(*list(resnet.children())[:-2])  # up to layer4
        self.resnet_pool = resnet.avgpool
        self.resnet_feat_dim = resnet.fc.in_features  # 2048

        self.ocr_engine = CnOcr(
            det_model_name=ocr_detector_name,
            rec_model_name=ocr_model_name,
            context=ocr_context,
        )

        vocab = self._extract_vocab(self.ocr_engine)
        special_tokens = ["<pad>", "<unk>"]
        unique_chars: List[str] = []
        for char in vocab:
            if not char:
                continue
            if char in special_tokens:
                continue
            if char not in unique_chars:
                unique_chars.append(char)

        self.idx_to_char = special_tokens + unique_chars
        self.char_to_idx = {char: idx for idx, char in enumerate(self.idx_to_char)}
        self.pad_idx = self.char_to_idx["<pad>"]
        self.unk_idx = self.char_to_idx["<unk>"]
        self.max_ocr_tokens = ocr_max_tokens

        self.ocr_embedding = nn.Embedding(
            num_embeddings=len(self.idx_to_char),
            embedding_dim=ocr_embedding_dim,
            padding_idx=self.pad_idx,
        )
        self.ocr_projection = nn.Sequential(
            nn.LayerNorm(ocr_embedding_dim),
            nn.Linear(ocr_embedding_dim, ocr_embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.ocr_feat_dim = ocr_embedding_dim

        self.fusion_dim = max(ocr_embedding_dim, 512)
        self.vision_token_proj = nn.Linear(self.resnet_feat_dim, self.fusion_dim)
        self.vision_global_proj = nn.Linear(self.resnet_feat_dim, self.fusion_dim)
        self.text_token_proj = nn.Linear(self.ocr_feat_dim, self.fusion_dim)

        self.cross_attn_v2t = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=8,
            batch_first=True,
        )
        self.cross_attn_t2v = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=8,
            batch_first=True,
        )

        self.vision_norm1 = nn.LayerNorm(self.fusion_dim)
        self.vision_norm2 = nn.LayerNorm(self.fusion_dim)
        self.text_norm1 = nn.LayerNorm(self.fusion_dim)
        self.text_norm2 = nn.LayerNorm(self.fusion_dim)
        self.vision_ffn = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
        )
        self.text_ffn = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
        )
        self.vision_global_norm = nn.LayerNorm(self.fusion_dim)
        self.text_global_norm = nn.LayerNorm(self.fusion_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.Sigmoid(),
        )

        # Normalization statistics to invert dataset preprocessing.
        mean = torch.tensor(image_mean or [0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor(image_std or [0.229, 0.224, 0.225], dtype=torch.float32)
        self.register_buffer("pixel_mean", mean.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", std.view(1, 3, 1, 1), persistent=False)

        self.feature_dim = self.fusion_dim * 4
        self.classifier = self._make_head(num_classes, dropout)
        self.dropout = dropout

        if freeze_resnet_backbone:
            self._freeze_module(self.resnet_conv)
            self._freeze_module(self.resnet_pool)
        if freeze_vision_backbone:
            warnings.warn(
                "freeze_vision_backbone is ignored when using the lightweight OCR branch.",
                stacklevel=2,
            )

    def _make_head(self, num_classes: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),
        )

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False

    def reset_classifier(self, num_classes: int, dropout: Optional[float] = None) -> None:
        """Replace the classification head (mirrors torchvision API)."""
        if dropout is None:
            dropout = self.dropout
        self.classifier = self._make_head(num_classes, dropout)

    def _extract_resnet_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_feats = self.resnet_conv(x)
        token_feats = conv_feats.flatten(2).transpose(1, 2)  # (B, HW, C)
        pooled = self.resnet_pool(conv_feats).flatten(1)
        return pooled, token_feats

    def _tensor_to_pil_list(self, x: torch.Tensor) -> List:
        """Unnormalize tensors and convert each item in the batch to PIL.Image."""
        # x is expected to be normalized via (x - mean) / std
        with torch.no_grad():
            denorm = x * self.pixel_std.to(x.device) + self.pixel_mean.to(x.device)
            denorm = denorm.clamp(0.0, 1.0)
            pil_images = [to_pil_image(img.cpu()) for img in denorm]
        return pil_images

    def _extract_ocr_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pil_images = self._tensor_to_pil_list(x)
        embed_device = self.ocr_embedding.weight.device
        token_sequences: List[torch.Tensor] = []

        for pil_image in pil_images:
            np_img = np.array(pil_image.convert("RGB"))

            try:
                ocr_result = self.ocr_engine.ocr(np_img)
            except TypeError:
                # Some CnOcr versions expect grayscale arrays for single-line mode.
                ocr_result = self.ocr_engine.ocr_for_single_line(
                    np.array(pil_image.convert("L"))
                )
            except Exception:
                ocr_result = ""

            text = self._flatten_ocr_result(ocr_result)
            token_ids = self._text_to_tokens(text)

            token_sequences.append(token_ids)

        batch_size = len(token_sequences)
        max_len = max((seq.numel() for seq in token_sequences), default=0)
        if max_len == 0:
            max_len = 1
        padded = torch.full(
            (batch_size, max_len),
            self.pad_idx,
            dtype=torch.long,
            device=embed_device,
        )
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=embed_device)

        for idx, seq in enumerate(token_sequences):
            length = seq.numel()
            if length == 0:
                continue
            padded[idx, :length] = seq
            mask[idx, :length] = True

        embeddings = self.ocr_embedding(padded)
        embeddings = self.ocr_projection(embeddings)
        return embeddings.to(x.device), mask.to(x.device)

    @staticmethod
    def _flatten_ocr_result(result: Union[str, Sequence, dict, None]) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return str(result.get("text", ""))
        if isinstance(result, Sequence):
            return "".join(ResNetOCRHybrid._flatten_ocr_result(item) for item in result)
        return str(result)

    def _text_to_tokens(self, text: str) -> torch.Tensor:
        if not text:
            return torch.empty(0, dtype=torch.long, device=self.ocr_embedding.weight.device)

        trimmed = text[: self.max_ocr_tokens]
        indices = [
            self.char_to_idx.get(char, self.unk_idx)
            for char in trimmed
        ]
        return torch.tensor(indices, dtype=torch.long, device=self.ocr_embedding.weight.device)

    @staticmethod
    def _extract_vocab(ocr_engine: CnOcr) -> Sequence[str]:
        vocab = getattr(ocr_engine, "vocab", None)
        if isinstance(vocab, dict):
            return list(vocab.keys())
        if isinstance(vocab, (list, tuple)):
            return list(vocab)
        get_vocab = getattr(ocr_engine, "get_vocab", None)
        if callable(get_vocab):
            extracted = get_vocab()
            if isinstance(extracted, dict):
                return list(extracted.keys())
            if isinstance(extracted, (list, tuple)):
                return list(extracted)
        # Fallback: basic alphanumeric set for unseen characters.
        return list(
            "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )

    @staticmethod
    def _masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return tensor.mean(dim=1)
        mask = mask.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (tensor * mask).sum(dim=1) / denom

    # ------------------------------------------------------------------ #
    # Main forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resnet_global, resnet_tokens = self._extract_resnet_features(x)
        ocr_tokens, ocr_mask = self._extract_ocr_features(x)

        vision_tokens = self.vision_token_proj(resnet_tokens)
        text_tokens = self.text_token_proj(ocr_tokens)

        text_padding_mask: Optional[torch.Tensor] = None
        if ocr_mask is not None and ocr_mask.any():
            text_padding_mask = ~ocr_mask

        cross_vis, _ = self.cross_attn_v2t(
            vision_tokens,
            text_tokens,
            text_tokens,
            key_padding_mask=text_padding_mask,
            need_weights=False,
        )
        vision_tokens = self.vision_norm1(vision_tokens + cross_vis)
        vision_tokens = self.vision_norm2(vision_tokens + self.vision_ffn(vision_tokens))

        cross_text, _ = self.cross_attn_t2v(
            text_tokens,
            vision_tokens,
            vision_tokens,
            need_weights=False,
        )
        text_tokens = self.text_norm1(text_tokens + cross_text)
        text_tokens = self.text_norm2(text_tokens + self.text_ffn(text_tokens))

        vision_context = vision_tokens.mean(dim=1)
        vision_global = self.vision_global_proj(resnet_global)
        vision_global = self.vision_global_norm(vision_global + vision_context)

        text_global = self._masked_mean(text_tokens, ocr_mask)
        text_global = self.text_global_norm(text_global)

        gate = self.fusion_gate(torch.cat([vision_global, text_global], dim=1))
        gated_mix = gate * vision_global + (1.0 - gate) * text_global
        interaction = vision_global * text_global

        fused = torch.cat(
            [vision_global, text_global, gated_mix, interaction],
            dim=1,
        )
        return self.classifier(fused)
