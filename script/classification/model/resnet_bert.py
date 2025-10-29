from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.mlp(self.avg_pool(x)).unsqueeze(-1).unsqueeze(-1)
        return x * weights


class FineGrainedResNet(nn.Module):
    """Dual-branch head on top of a ResNet-50 backbone for fine-grained cues."""

    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.mid_attn = ChannelAttention(1024)
        self.high_attn = ChannelAttention(2048)
        self.mid_pool = nn.AdaptiveAvgPool2d(1)
        self.high_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 1024 + 2048
        self.dropout = dropout
        self.classifier = self._make_head(num_classes, dropout)

    def _make_head(self, num_classes: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),
        )

    def reset_classifier(self, num_classes: int, dropout: float) -> None:
        self.dropout = dropout
        self.classifier = self._make_head(num_classes, dropout)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        mid = self.layer3(x)
        high = self.layer4(mid)

        mid = self.mid_attn(mid)
        high = self.high_attn(high)
        mid = self.mid_pool(mid).flatten(1)
        high = self.high_pool(high).flatten(1)
        feats = torch.cat([mid, high], dim=1)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.classifier(feats)


class FineGrainedResNetTextFusion(nn.Module):
    """Fine-grained image encoder fused with a BERT-based text encoder."""

    def __init__(
        self,
        num_classes: int = 1000,
        pretrained_image: bool = True,
        dropout: float = 0.3,
        text_model_name: str = "bert-base-chinese",
        text_trainable: bool = False,
        fusion_dim: int = 768,
        max_text_length: int = 128,
    ) -> None:
        super().__init__()

        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "FineGrainedResNetTextFusion requires the 'transformers' package. "
                "Install it via `pip install transformers`."
            ) from exc

        self.image_encoder = FineGrainedResNet(num_classes=1, pretrained=pretrained_image, dropout=dropout)
        self.image_encoder.classifier = nn.Identity()
        self.image_feature_dim = self.image_encoder.feature_dim

        config = AutoConfig.from_pretrained(text_model_name)
        config.max_position_embeddings = max(config.max_position_embeddings, max_text_length)
        self.text_model = AutoModel.from_pretrained(text_model_name, config=config)
        self.text_hidden_dim = int(getattr(self.text_model.config, "hidden_size", fusion_dim))
        self.max_text_length = max_text_length

        if not text_trainable:
            for parameter in self.text_model.parameters():
                parameter.requires_grad = False

        self.image_proj = nn.Sequential(
            nn.LayerNorm(self.image_feature_dim),
            nn.Linear(self.image_feature_dim, fusion_dim),
            nn.GELU(),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(self.text_hidden_dim),
            nn.Linear(self.text_hidden_dim, fusion_dim),
            nn.GELU(),
        )

        self.fusion_dim = fusion_dim
        self.fused_dim = fusion_dim * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.fused_dim),
            nn.Linear(self.fused_dim, self.fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.fused_dim),
            nn.Linear(self.fused_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_dim, num_classes),
        )

        self.num_classes = num_classes
        self.dropout = dropout

    def reset_classifier(self, num_classes: int, dropout: Optional[float] = None) -> None:
        if dropout is None:
            dropout = self.dropout
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.fused_dim),
            nn.Linear(self.fused_dim, self.fusion_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.fusion_dim, num_classes),
        )

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        image_feats = self.image_encoder.forward_features(image)
        image_embed = self.image_proj(image_feats)

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled = text_outputs.pooler_output
        if pooled is None:
            pooled = text_outputs.last_hidden_state[:, 0]
        text_embed = self.text_proj(pooled)

        fused = torch.cat([image_embed, text_embed], dim=1)
        fused = self.fusion(fused)
        return self.classifier(fused)


__all__ = ["FineGrainedResNet", "FineGrainedResNetTextFusion"]
