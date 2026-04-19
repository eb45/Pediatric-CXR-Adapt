"""
DANN-style domain adaptation for CXR (Ganin et al., arXiv:1505.07818).

Use during pediatric fine-tuning: jointly minimize pathology CLIP loss on both domains
and train a domain discriminator on image embeddings; gradient reversal makes the image
encoder learn domain-invariant features.

Typical use: after adult contrastive pretraining, call :func:`finetune_pediatric_clip_dann`
with the same ``ImageTextModel``, frozen text prompts, pediatric train loader, and an
adult train loader (both manifests have binary path / label columns).
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from cxr_model import ImageTextModel, clip_style_loss


class _GradientReversalFn(torch.autograd.Function):
    """Forward: identity. Backward: multiply gradient by ``-alpha``."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Gradient reversal layer; ``alpha`` is the DANN λ (can schedule per epoch)."""
    return _GradientReversalFn.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    """MLP on image embedding (same dim as CLIP space). Predicts domain: 0=adult, 1=pediatric."""

    def __init__(self, embed_dim: int = 512, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Logits [B] for P(pediatric)."""
        return self.net(z).squeeze(-1)


def _cycle_loader(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def finetune_pediatric_clip_dann(
    model: ImageTextModel,
    ped_loader: DataLoader,
    adult_loader: DataLoader,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    domain_lambda: float = 0.1,
    weight_decay: float = 1e-4,
    lambda_schedule: Optional[str] = None,
) -> torch.Tensor:
    """
    Pediatric fine-tuning with DANN: freeze BERT; train image encoder + logit_scale + domain head.

    Each step pairs one pediatric mini-batch with one adult mini-batch (same sizes if loaders match).
    Domain label: 1 = pediatric, 0 = adult.

    **domain_lambda**: weight on domain adversarial loss (γ in many papers).
    **lambda_schedule**: if ``"linear"``, ramps λ from 0 → ``domain_lambda`` over epochs.
    Returns frozen text embeddings ``[2, D]`` (same as :func:`cxr_engine.finetune_pediatric_clip`).
    """
    for p in model.text_encoder.parameters():
        p.requires_grad = False
    with torch.no_grad():
        text_emb = model.encode_text_batch(input_ids, attention_mask).detach()

    embed_dim = text_emb.shape[1]
    domain_disc = DomainDiscriminator(embed_dim=embed_dim).to(device)

    train_params = (
        list(model.image_encoder.parameters())
        + [model.logit_scale]
        + list(domain_disc.parameters())
    )
    opt = torch.optim.AdamW(train_params, lr=lr, weight_decay=weight_decay)

    adult_cycle = _cycle_loader(adult_loader)
    model.train()
    domain_disc.train()

    n_batches = len(ped_loader)
    for ep in range(epochs):
        if lambda_schedule == "linear":
            lam = domain_lambda * float(ep + 1) / float(max(epochs, 1))
        else:
            lam = domain_lambda

        pbar = tqdm(ped_loader, desc=f"DANN pediatric ft ep{ep + 1}/{epochs} λ={lam:.4f}")
        for x_p, y_p in pbar:
            x_a, y_a = next(adult_cycle)

            x_p, y_p = x_p.to(device), y_p.to(device)
            x_a, y_a = x_a.to(device), y_a.to(device)

            # Match batch sizes (take minimum)
            n = min(x_p.shape[0], x_a.shape[0])
            if n == 0:
                continue
            x_p, y_p = x_p[:n], y_p[:n]
            x_a, y_a = x_a[:n], y_a[:n]

            x = torch.cat([x_p, x_a], dim=0)
            y = torch.cat([y_p, y_a], dim=0)
            domain = torch.cat(
                [torch.ones(n, device=device), torch.zeros(n, device=device)],
                dim=0,
            )

            opt.zero_grad()
            z = model.encode_image(x)
            loss_cls = clip_style_loss(z, text_emb, y, model.logit_scale)

            z_rev = grad_reverse(z, lam)
            d_logits = domain_disc(z_rev)
            loss_dom = F.binary_cross_entropy_with_logits(d_logits, domain)

            loss = loss_cls + loss_dom
            loss.backward()
            opt.step()

            pbar.set_postfix(
                cls=f"{loss_cls.item():.4f}",
                dom=f"{loss_dom.item():.4f}",
            )

    domain_disc.eval()
    return text_emb


@torch.no_grad()
def domain_accuracy(
    model: ImageTextModel,
    domain_disc: DomainDiscriminator,
    ped_loader: DataLoader,
    adult_loader: DataLoader,
    device: torch.device,
) -> float:
    """Fraction of samples where domain discriminator predicts correct domain (sanity check)."""
    model.eval()
    domain_disc.eval()
    correct, total = 0, 0
    adult_cycle = _cycle_loader(adult_loader)
    for x_p, _ in ped_loader:
        x_a, _ = next(adult_cycle)
        x_p = x_p.to(device)
        x_a = x_a.to(device)
        n = min(x_p.shape[0], x_a.shape[0])
        if n == 0:
            continue
        x = torch.cat([x_p[:n], x_a[:n]], dim=0)
        dom = torch.cat([torch.ones(n, device=device), torch.zeros(n, device=device)])
        z = model.encode_image(x)
        pred = (torch.sigmoid(domain_disc(z)) >= 0.5).float()
        correct += (pred == dom).sum().item()
        total += dom.numel()
    return correct / max(total, 1)
