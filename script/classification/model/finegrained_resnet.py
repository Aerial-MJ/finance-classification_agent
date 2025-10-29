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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.classifier(feats)


__all__ = ["FineGrainedResNet"]
