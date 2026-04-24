"""
Dual image–text encoder (CLIP-style) with BERT text tower and ViT/ResNet image tower.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        embed_dim: int = 512,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        if backbone.startswith("resnet"):
            from torchvision import models

            if backbone == "resnet18":
                w = models.ResNet18_Weights.DEFAULT if pretrained else None
                net = models.resnet18(weights=w)
                in_f = net.fc.in_features
                net.fc = nn.Identity()
            elif backbone == "resnet50":
                w = models.ResNet50_Weights.DEFAULT if pretrained else None
                net = models.resnet50(weights=w)
                in_f = net.fc.in_features
                net.fc = nn.Identity()
            else:
                raise ValueError(f"Unknown backbone {backbone}")
            self.trunk = net
        elif backbone.startswith("vit"):
            from torchvision import models

            if backbone == "vit_b_16":
                w = models.ViT_B_16_Weights.DEFAULT if pretrained else None
                net = models.vit_b_16(weights=w)
                in_f = net.heads.head.in_features
                net.heads.head = nn.Identity()
            else:
                raise ValueError(f"Unknown backbone {backbone}")
            self.trunk = net
        else:
            raise ValueError(f"Unknown backbone {backbone}")

        self.proj = nn.Sequential(
            nn.Linear(in_f, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(x)
        z = self.proj(feat)
        return l2_normalize(z, dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, bert_model: str = "bert-base-uncased", embed_dim: int = 512) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        hid = self.bert.config.hidden_size
        self.proj = nn.Linear(hid, embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        z = self.proj(cls)
        return l2_normalize(z, dim=-1)


class ImageTextModel(nn.Module):
    """CLIP-style model: image and text embeddings in a shared space."""

    def __init__(
        self,
        image_backbone: str = "resnet18",
        bert_model: str = "bert-base-uncased",
        embed_dim: int = 512,
        pretrained_image: bool = True,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(image_backbone, embed_dim, pretrained=pretrained_image)
        self.text_encoder = TextEncoder(bert_model, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592)) 

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def encode_text_batch(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.text_encoder(input_ids, attention_mask)

    def contrastive_logits(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        image_emb: [B, D], text_emb: [K, D] (K class prompts).
        Returns logits [B, K].
        """
        logit_scale = self.logit_scale.exp().clamp(1, 100)
        return logit_scale * image_emb @ text_emb.t()


def clip_style_loss(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Supervised CLIP loss: softmax over class text embeddings (CheXzero-style).
    image_emb: [B,D], text_emb: [C,D], labels: [B] with values in 0..C-1.
    """
    logits = logit_scale.exp().clamp(1, 100) * image_emb @ text_emb.t()
    return F.cross_entropy(logits, labels)


def info_nce_symmetric(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    When there are exactly 2 class prompts, supervised CE is equivalent to a 2-way InfoNCE
    with positives defined by class. This function keeps the same CE for clarity.
    """
    return clip_style_loss(image_emb, text_emb, labels, logit_scale)
