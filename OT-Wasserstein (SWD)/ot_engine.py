"""
Training and evaluation engine for Wasserstein domain adaptation.

Mirrors cxr_engine.py API so callers (notebooks, run scripts) can swap
the two engines with minimal changes. Re-uses:
  - cxr_engine.{BaselineClassifier, build_transform, load_prompt_tensors,
                evaluate_baseline, EvalResult, stratified_subset_indices,
                prepare_loaders_from_manifests, train_baseline}
  - cxr_data.*
  - cxr_eval_viz.*  (all evaluation figures)

New public API:
  train_ot_domain_adapt(...)           Adult pre-train with OT alignment
  finetune_ot_pediatric(...)           Pediatric fine-tune on OT model
  evaluate_ot_classifier(...)          Evaluate WassersteinDA model (same signature as evaluate_clip_classifier)
  ot_learning_curve_experiment(...)    Same signature as learning_curve_experiment
  save_ot_bundle / load_ot_bundle      Checkpoint helpers
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


from cxr_data import ManifestImageDataset
from cxr_engine import (
    BaselineClassifier,
    EvalResult,
    build_transform,
    evaluate_baseline,
    load_prompt_tensors,
    prepare_loaders_from_manifests,
    stratified_subset_indices,
    train_baseline,
)
from ot_model import WassersteinDomainAdaptationModel, build_ot_model


def train_ot_domain_adapt(
    model: WassersteinDomainAdaptationModel,
    adult_loader: DataLoader,
    ped_train_loader: DataLoader,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float = 1e-4,
    reversal_schedule: str = "linear",
    freeze_text: bool = True,
    verbose: bool = True,
) -> List[Dict[str, float]]:
    """
    Joint adult CLIP + OT alignment pre-training.

    Pairs adult and pediatric batches on-the-fly (cycle the shorter loader).
    Text encoder is frozen by default (same design decision as train_adult_clip).

    Args:
        reversal_schedule: 'linear' (alpha ramps from 0→1) or 'constant' (alpha=1).
        freeze_text:        Freeze entire BERT text tower (recommended).

    Returns:
        List of per-epoch loss dicts for logging.
    """
    if freeze_text:
        for p in model.text_encoder.parameters():
            p.requires_grad = False
        model.text_encoder.eval()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(adult_loader)
    step = 0
    history: List[Dict[str, float]] = []

    ped_iter = iter(ped_train_loader)

    for ep in range(epochs):
        model.train()
        if freeze_text:
            model.text_encoder.eval()

        ep_total = ep_clip = ep_ot = ep_adv = 0.0
        n_batches = 0

        for adult_imgs, adult_labels in tqdm(adult_loader, desc=f"OT ep{ep+1}/{epochs}", disable=not verbose):
            try:
                ped_imgs, ped_labels = next(ped_iter)
            except StopIteration:
                ped_iter = iter(ped_train_loader)
                ped_imgs, ped_labels = next(ped_iter)

            adult_imgs = adult_imgs.to(device)
            adult_labels = adult_labels.to(device)
            ped_imgs = ped_imgs.to(device)
            ped_labels = ped_labels.to(device)

            if reversal_schedule == "linear":
                alpha = min(1.0, step / max(1, total_steps - 1))
            else:
                alpha = 1.0

            opt.zero_grad()
            out = model(
                adult_imgs,
                adult_labels,
                ped_imgs,
                input_ids,
                attention_mask,
                ped_labels=ped_labels,
                reversal_alpha=float(alpha),
            )
            out.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            opt.step()

            ep_total += out.total_loss.item()
            ep_clip += out.clip_loss.item()
            ep_ot += out.ot_loss.item()
            ep_adv += out.adversarial_loss.item()
            n_batches += 1
            step += 1

        log = {
            "epoch": ep + 1,
            "loss": ep_total / max(1, n_batches),
            "clip_loss": ep_clip / max(1, n_batches),
            "ot_loss": ep_ot / max(1, n_batches),
            "adv_loss": ep_adv / max(1, n_batches),
        }
        if verbose:
            print(
                f"  ep{ep+1}/{epochs}  loss={log['loss']:.4f}  "
                f"clip={log['clip_loss']:.4f}  ot={log['ot_loss']:.4f}  "
                f"adv={log['adv_loss']:.4f}"
            )
        history.append(log)

    return history


def finetune_ot_pediatric(
    model: WassersteinDomainAdaptationModel,
    ped_loader: DataLoader,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float = 1e-4,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Fine-tune image encoder + logit_scale on labeled pediatric data.
    Text encoder stays frozen (same as finetune_pediatric_clip).

    Returns frozen text embeddings [2, D].
    """
    for p in model.text_encoder.parameters():
        p.requires_grad = False
    model.text_encoder.eval()

    with torch.no_grad():
        text_emb = model.encode_text_batch(input_ids, attention_mask).detach()

    train_params = list(model.image_encoder.parameters()) + [model.logit_scale]
    opt = torch.optim.AdamW(train_params, lr=lr, weight_decay=weight_decay)

    model.train()
    model.text_encoder.eval()

    for ep in range(epochs):
        ep_loss = 0.0
        n = 0
        for x, y in tqdm(ped_loader, desc=f"PED ft ep{ep+1}/{epochs}", disable=not verbose):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            z = model.encode_image(x)
            scale = model.logit_scale.exp().clamp(1, 100)
            logits = scale * (z @ text_emb.T)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(y)
            n += len(y)
        if verbose:
            print(f"  ped ft ep{ep+1}/{epochs}  loss={ep_loss/max(1,n):.4f}")

    return text_emb


@torch.no_grad()
def evaluate_ot_classifier(
    model: WassersteinDomainAdaptationModel,
    loader: DataLoader,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    frozen_text_emb: Optional[torch.Tensor] = None,
) -> EvalResult:
    """
    Drop-in replacement for cxr_engine.evaluate_clip_classifier.
    Works with WassersteinDomainAdaptationModel.
    """
    model.eval()
    if frozen_text_emb is not None:
        text_emb = frozen_text_emb
    else:
        text_emb = model.encode_text_batch(input_ids, attention_mask)

    probs_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        z = model.encode_image(x)
        scale = model.logit_scale.exp().clamp(1, 100)
        logits = scale * (z @ text_emb.T)
        p = F.softmax(logits, dim=1)[:, 1]
        probs_list.append(p.cpu().numpy())
        y_list.append(y.numpy())

    probs = np.concatenate(probs_list)
    labels = np.concatenate(y_list).astype(np.int64)
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")
    pred = (probs >= 0.5).astype(np.int64)
    f1 = float(f1_score(labels, pred, zero_division=0))
    return EvalResult(auc=auc, f1=f1, probs_positive=probs, labels=labels)



def ot_learning_curve_experiment(
    adult_train_loader: DataLoader,
    ped_train_dataset: ManifestImageDataset,
    ped_test_loader: DataLoader,
    prompt_path: Path,
    device: torch.device,
    fractions: List[float],
    adult_epochs: int,
    ped_epochs: int,
    lr_adult: float,
    lr_ped: float,
    image_backbone: str,
    baseline_epochs: int,
    lr_base: float,
    batch_size: int,
    seed: int,
    # OT-specific
    lambda_ot: float = 0.5,
    lambda_adv: float = 0.1,
    lambda_ped: float = 1.0,
    n_projections: int = 256,
    verbose: bool = True,
) -> Dict[str, List[Dict[str, float]]]:
    """
    Pre-trains the OT model once on the full adult loader + full pediatric
    train set, then sweeps fractions for fine-tuning (same protocol as
    cxr_engine.learning_curve_experiment).

    Returns dict with keys 'proposed' (OT) and 'baseline'.
    """
    input_ids, attention_mask = load_prompt_tensors(prompt_path, device)

    # temp full pediatric loader for joint pre-training 
    ped_full_loader = DataLoader(
        ped_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = build_ot_model(
        image_backbone=image_backbone,
        lambda_ot=lambda_ot,
        lambda_adv=lambda_adv,
        lambda_ped=lambda_ped,
        n_projections=n_projections,
        pretrained_image=True,
    ).to(device)

    if verbose:
        print(f"[OT] Pre-training ({adult_epochs} epochs, backbone={image_backbone}, "
              f"λ_ot={lambda_ot}, n_proj={n_projections})")
    train_ot_domain_adapt(
        model,
        adult_train_loader,
        ped_full_loader,
        input_ids,
        attention_mask,
        device,
        epochs=adult_epochs,
        lr=lr_adult,
        verbose=verbose,
    )
    adult_ckpt = copy.deepcopy(model.state_dict())
    del model

    labels = np.array(ped_train_dataset.labels)
    results_ot: List[Dict[str, float]] = []
    results_baseline: List[Dict[str, float]] = []

    for frac in fractions:
        idx = stratified_subset_indices(labels, frac, seed)
        sub = Subset(ped_train_dataset, idx.tolist())
        ped_loader = DataLoader(sub, batch_size=batch_size, shuffle=True, num_workers=0)

        m = build_ot_model(
            image_backbone=image_backbone,
            lambda_ot=lambda_ot,
            lambda_adv=lambda_adv,
            lambda_ped=lambda_ped,
            n_projections=n_projections,
            pretrained_image=True,
        ).to(device)
        m.load_state_dict(adult_ckpt)
        ft = finetune_ot_pediatric(
            m, ped_loader, input_ids, attention_mask, device, ped_epochs, lr_ped, verbose=verbose
        )
        ev = evaluate_ot_classifier(m, ped_test_loader, input_ids, attention_mask, device, frozen_text_emb=ft)
        results_ot.append({"fraction": float(frac), "auc": ev.auc, "f1": ev.f1})
        if verbose:
            print(f"  [OT frac={frac:.2f}] AUC={ev.auc:.4f}  F1={ev.f1:.4f}")
        del m

        # baseline
        torch.manual_seed(seed + int(1000 * frac))
        bl = train_baseline(ped_loader, None, device, baseline_epochs, lr_base,
                            backbone="resnet18", pretrained=True)
        evb = evaluate_baseline(bl, ped_test_loader, device)
        results_baseline.append({"fraction": float(frac), "auc": evb.auc, "f1": evb.f1})
        if verbose:
            print(f"  [Baseline frac={frac:.2f}] AUC={evb.auc:.4f}  F1={evb.f1:.4f}")

    return {"proposed": results_ot, "baseline": results_baseline}



def save_ot_bundle(
    path: Path,
    model: WassersteinDomainAdaptationModel,
    frozen_text_emb: torch.Tensor,
    *,
    image_backbone: str,
    image_size: int,
    n_projections: int,
    lambda_ot: float,
    lambda_adv: float,
    lambda_ped: float,
    batch_size: int = 16,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "frozen_text_emb": frozen_text_emb.detach().cpu(),
            "image_backbone": str(image_backbone),
            "embed_dim": model.embed_dim,
            "image_size": int(image_size),
            "n_projections": int(n_projections),
            "lambda_ot": float(lambda_ot),
            "lambda_adv": float(lambda_adv),
            "lambda_ped": float(lambda_ped),
            "batch_size": int(batch_size),
        },
        path,
    )
    print(f"Saved OT bundle -> {path}")
