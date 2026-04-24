"""
Optimal Transport (Sliced Wasserstein) domain adaptation for pediatric CXR.

Architecture
------------
Reuses ImageEncoder and TextEncoder from cxr_model.py unchanged.

Adds:
  • SlicedWassersteinLoss — correct O(n log n) approximation of W2 using
    random 1-D projections (no LP solver, fully differentiable). Handles
    unequal batch sizes via proper quantile-function interpolation.

  • WassersteinDomainAdaptationModel — shared ImageEncoder aligned across
    adult (source) and pediatric (target) domains via SWD, on top of the
    CLIP-style classification loss.

  • WassersteinDomainAdaptationModel.forward() returns a WassersteinOutputs
    dataclass so callers can inspect each loss term independently.

Training objective
------------------
  L = L_clip_adult + λ_ped * L_clip_ped + λ_ot * SWD(z_adult, z_ped)
    + λ_adv * L_domain

Usage
-----
>>> from ot_model import WassersteinDomainAdaptationModel, build_ot_model
>>> model = build_ot_model(image_backbone="resnet18")
>>> model.to(device)
>>> out = model(adult_imgs, adult_labels, ped_imgs, ped_labels,
...             input_ids, attention_mask)
>>> out.total_loss.backward()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cxr_model import ImageEncoder, ImageTextModel, TextEncoder, l2_normalize


def _quantile_interp(x_sorted: torch.Tensor, target_n: int) -> torch.Tensor:
    """
    Resample a sorted empirical CDF of size N to size `target_n` by evaluating
    its quantile function (inverse CDF) at evenly spaced probability points.

    This is the statistically correct way to compare empirical distributions
    of different sizes in 1-D Wasserstein: we evaluate both quantile functions
    at the same set of probability points and compare them pointwise.

    Args:
        x_sorted:  [N, n_proj] — already sorted along dim 0.
        target_n:  Target number of quantile points M.

    Returns:
        [M, n_proj] quantile-interpolated values.
    """
    N = x_sorted.shape[0]
    if N == target_n:
        return x_sorted

    src_q = (torch.arange(N, device=x_sorted.device, dtype=x_sorted.dtype) + 0.5) / N
    tgt_q = (torch.arange(target_n, device=x_sorted.device, dtype=x_sorted.dtype) + 0.5) / target_n
    idx_hi = torch.searchsorted(src_q.contiguous(), tgt_q.contiguous())
    idx_hi = idx_hi.clamp(0, N - 1)
    idx_lo = (idx_hi - 1).clamp(0, N - 1)

    q_lo = src_q[idx_lo]   
    q_hi = src_q[idx_hi]   

    dq = (q_hi - q_lo).clamp(min=1e-12)
    w_hi = ((tgt_q - q_lo) / dq).clamp(0.0, 1.0)  
    w_lo = 1.0 - w_hi                               

 
    v_lo = x_sorted[idx_lo]  
    v_hi = x_sorted[idx_hi]  

    return w_lo.unsqueeze(1) * v_lo + w_hi.unsqueeze(1) * v_hi 


class SlicedWassersteinLoss(nn.Module):
    """
    Sliced Wasserstein-2 Distance between two empirical distributions.

    Algorithm
    ---------
    1. Sample `n_projections` random unit vectors θ on S^{D-1}.
    2. Project both sets of embeddings onto each θ: x·θ, y·θ  → 1-D samples.
    3. Sort each projected set (= empirical quantile function).
    4. If N ≠ M, evaluate both quantile functions at the same M evenly-spaced
       probability points via piecewise-linear interpolation (correct for W2;
       avoids the bias introduced by naive linear interpolation of sorted values).
    5. SWD² = mean over projections of the 1-D W2² distance.
       SWD  = sqrt(SWD²)  returned as the loss scalar.

    This is an unbiased estimator of W2(μ, ν) as n_projections → ∞.

    Args:
        n_projections: Number of random 1-D projections. 256 is recommended
                       for 512-dim embeddings to keep estimator variance low.
                       128 is acceptable for faster debugging runs.
        p:             Wasserstein order (1 or 2). Default 2.
    """

    def __init__(self, n_projections: int = 256, p: int = 2) -> None:
        super().__init__()
        if p not in (1, 2):
            raise ValueError(f"p must be 1 or 2, got {p}")
        self.n_projections = n_projections
        self.p = p

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, D] source embeddings (adult domain).
            y: [M, D] target embeddings (pediatric domain). N and M may differ.
        Returns:
            Scalar SWD estimate (W_p units, not W_p^p).
        """
        N, D = x.shape
        M    = y.shape[0]

        theta = torch.randn(D, self.n_projections, device=x.device, dtype=x.dtype)
        theta = F.normalize(theta, p=2, dim=0)

        x_proj = x @ theta
        y_proj = y @ theta

        x_sorted, _ = torch.sort(x_proj, dim=0)   
        y_sorted, _ = torch.sort(y_proj, dim=0)  


        common_n = max(N, M)
        x_q = _quantile_interp(x_sorted, common_n)  
        y_q = _quantile_interp(y_sorted, common_n) 


        if self.p == 1:
            w_per_proj = (x_q - y_q).abs().mean(dim=0)     
            return w_per_proj.mean()
        else:  
            w2_per_proj = ((x_q - y_q) ** 2).mean(dim=0)     
            swd2 = w2_per_proj.mean()                         
            return swd2.sqrt().clamp(min=1e-8)               



class GradientReversalFunction(torch.autograd.Function):
    """Reverses gradients during backward pass (used in adversarial domain loss)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:  
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    """
    Binary domain classifier: adult (0) vs pediatric (1).
    Used with gradient reversal for adversarial domain alignment.
    """

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Returns logits [N, 1]. alpha controls reversal strength."""
        x_rev = grad_reverse(x, alpha)
        return self.net(x_rev).squeeze(-1)  # [N]




@dataclass
class WassersteinOutputs:
    total_loss: torch.Tensor
    clip_loss: torch.Tensor         
    ot_loss: torch.Tensor            
    adversarial_loss: torch.Tensor    
    adult_class_loss: torch.Tensor  
    ped_class_loss: torch.Tensor     



class WassersteinDomainAdaptationModel(nn.Module):
    """
    CLIP-style ViT/ResNet + Wasserstein (OT) domain adaptation.

    Training objective:
        L_total = L_clip_adult + λ_ped * L_clip_ped + λ_ot * L_ot + λ_adv * L_adv

    where:
        L_clip_{adult,ped}  = supervised CLIP (cross-entropy over class prompts)
        L_ot                = Sliced Wasserstein Distance between adult/ped embeddings
        L_adv               = gradient-reversal adversarial domain loss (optional)

    Args:
        image_backbone:     'resnet18', 'resnet50', or 'vit_b_16'.
        bert_model:         HuggingFace BERT variant.
        embed_dim:          Shared embedding dimension.
        n_projections:      Number of SWD random projections (default 256).
        lambda_ot:          Weight for SWD alignment loss.
        lambda_adv:         Weight for adversarial domain loss (0 = disabled).
        lambda_ped:         Weight for pediatric CLIP loss.
        pretrained_image:   Load ImageNet pretrained weights.
    """

    def __init__(
        self,
        image_backbone: str = "resnet18",
        bert_model: str = "bert-base-uncased",
        embed_dim: int = 512,
        n_projections: int = 256,
        lambda_ot: float = 0.5,
        lambda_adv: float = 0.1,
        lambda_ped: float = 1.0,
        pretrained_image: bool = True,
    ) -> None:
        super().__init__()
  
        self.image_encoder = ImageEncoder(image_backbone, embed_dim, pretrained=pretrained_image)
        self.text_encoder = TextEncoder(bert_model, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

        self.ot_loss_fn = SlicedWassersteinLoss(n_projections=n_projections, p=2)

        self.domain_discriminator = DomainDiscriminator(embed_dim=embed_dim)

        self.lambda_ot = lambda_ot
        self.lambda_adv = lambda_adv
        self.lambda_ped = lambda_ped
        self.embed_dim = embed_dim


    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(x)

    def encode_text_batch(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.text_encoder(input_ids, attention_mask)

    def contrastive_logits(
        self, image_emb: torch.Tensor, text_emb: torch.Tensor
    ) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(1, 100)
        return scale * image_emb @ text_emb.T


    def _clip_loss(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.contrastive_logits(image_emb, text_emb)
        return F.cross_entropy(logits, labels)

    def _domain_loss(
        self,
        adult_emb: torch.Tensor,
        ped_emb: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """BCE loss on domain classifier with gradient reversal."""
        all_emb = torch.cat([adult_emb, ped_emb], dim=0)
        domain_labels = torch.cat(
            [
                torch.zeros(adult_emb.shape[0], device=adult_emb.device),
                torch.ones(ped_emb.shape[0], device=ped_emb.device),
            ],
            dim=0,
        )
        logits = self.domain_discriminator(all_emb, alpha=alpha)
        return F.binary_cross_entropy_with_logits(logits, domain_labels)


    def forward(
        self,
        adult_imgs: torch.Tensor,
        adult_labels: torch.Tensor,
        ped_imgs: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ped_labels: Optional[torch.Tensor] = None,
        reversal_alpha: float = 1.0,
    ) -> WassersteinOutputs:
        """
        Args:
            adult_imgs:       [B_a, 3, H, W]  adult source images.
            adult_labels:     [B_a]            0/1 labels for adult images.
            ped_imgs:         [B_p, 3, H, W]  pediatric target images.
            input_ids:        [2, L]           BERT token ids for 2 class prompts.
            attention_mask:   [2, L]           BERT attention mask.
            ped_labels:       [B_p] or None    Pediatric labels (None = unlabelled target).
            reversal_alpha:   Gradient reversal magnitude.

        Returns:
            WassersteinOutputs with .total_loss and per-component losses.
        """

        with torch.no_grad():
            text_emb = self.text_encoder(input_ids, attention_mask)


        adult_emb = self.image_encoder(adult_imgs)   
        ped_emb = self.image_encoder(ped_imgs)      


        adult_class_loss = self._clip_loss(adult_emb, text_emb, adult_labels)

   
        if ped_labels is not None:
            ped_class_loss = self._clip_loss(ped_emb, text_emb, ped_labels)
        else:
            ped_class_loss = torch.tensor(0.0, device=adult_imgs.device)

        clip_loss = adult_class_loss + self.lambda_ped * ped_class_loss

        ot_loss = self.ot_loss_fn(adult_emb, ped_emb)

        if self.lambda_adv > 0:
            adv_loss = self._domain_loss(adult_emb, ped_emb, alpha=reversal_alpha)
        else:
            adv_loss = torch.tensor(0.0, device=adult_imgs.device)

        total_loss = clip_loss + self.lambda_ot * ot_loss + self.lambda_adv * adv_loss

        return WassersteinOutputs(
            total_loss=total_loss,
            clip_loss=clip_loss,
            ot_loss=ot_loss,
            adversarial_loss=adv_loss,
            adult_class_loss=adult_class_loss,
            ped_class_loss=ped_class_loss,
        )

def build_ot_model(
    image_backbone: str = "resnet18",
    embed_dim: int = 512,
    n_projections: int = 256,
    lambda_ot: float = 0.5,
    lambda_adv: float = 0.1,
    lambda_ped: float = 1.0,
    pretrained_image: bool = True,
    **kwargs,
) -> WassersteinDomainAdaptationModel:
    """Convenience constructor with sensible defaults."""
    return WassersteinDomainAdaptationModel(
        image_backbone=image_backbone,
        embed_dim=embed_dim,
        n_projections=n_projections,
        lambda_ot=lambda_ot,
        lambda_adv=lambda_adv,
        lambda_ped=lambda_ped,
        pretrained_image=pretrained_image,
        **kwargs,
    )


def load_ot_bundle(
    path: str,
    device: torch.device,
) -> Tuple[WassersteinDomainAdaptationModel, torch.Tensor, dict]:
    """
    Load a checkpoint saved by ot_engine.save_ot_bundle().
    Returns (model, frozen_text_emb, meta).
    """
    ckpt = torch.load(path, map_location=device)
    model = build_ot_model(
        image_backbone=ckpt["image_backbone"],
        embed_dim=ckpt.get("embed_dim", 512),
        n_projections=ckpt.get("n_projections", 256),
        lambda_ot=ckpt.get("lambda_ot", 0.5),
        lambda_adv=ckpt.get("lambda_adv", 0.1),
        lambda_ped=ckpt.get("lambda_ped", 1.0),
        pretrained_image=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    text_emb = ckpt["frozen_text_emb"].to(device)
    meta = {k: v for k, v in ckpt.items() if k not in ("model_state_dict", "frozen_text_emb")}
    return model, text_emb, meta
