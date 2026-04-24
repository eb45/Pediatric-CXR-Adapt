"""
Training, evaluation (AUC-ROC, F1), t-SNE, and learning-curve sweeps for CXR domain adaptation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from tqdm import tqdm

from cxr_data import ManifestImageDataset, apply_remap_df, filter_existing, read_manifest_csv
from cxr_model import ImageTextModel, clip_style_loss


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )


def load_prompt_tensors(path: Path, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    bundle = torch.load(path, map_location=device)
    return bundle["input_ids"].to(device), bundle["attention_mask"].to(device)


@dataclass
class EvalResult:
    auc: float
    f1: float
    probs_positive: np.ndarray
    labels: np.ndarray


@torch.no_grad()
def evaluate_clip_classifier(
    model: ImageTextModel,
    loader: DataLoader,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    frozen_text_emb: Optional[torch.Tensor] = None,
) -> EvalResult:
    model.eval()
    if frozen_text_emb is not None:
        text_emb = frozen_text_emb
    else:
        text_emb = model.encode_text_batch(input_ids, attention_mask)

    probs_list: List[float] = []
    y_list: List[int] = []
    for x, y in loader:
        x = x.to(device)
        z = model.encode_image(x)
        logits = z @ text_emb.t() * model.logit_scale.exp().clamp(1, 100)
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


class BaselineClassifier(nn.Module):
    """Supervised baseline: ResNet trunk + linear head (pediatric only)."""

    def __init__(self, backbone: str = "resnet18", num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
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
            raise ValueError(backbone)
        self.trunk = net
        self.head = nn.Linear(in_f, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


def train_baseline(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    backbone: str = "resnet18",
    pretrained: bool = True,
) -> BaselineClassifier:
    model = BaselineClassifier(backbone=backbone, pretrained=pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                tot, n = 0.0, 0
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    tot += crit(model(x), y).item() * len(y)
                    n += len(y)
    return model


@torch.no_grad()
def evaluate_baseline(model: BaselineClassifier, loader: DataLoader, device: torch.device) -> EvalResult:
    model.eval()
    probs_list: List[float] = []
    y_list: List[int] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
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


def train_adult_clip(
    model: ImageTextModel,
    loader: DataLoader,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    text_encoder_train_mode: str = "frozen",
    weight_decay: float = 1e-4,
) -> None:
    """
    Adult CLIP-style pretraining.

    **text_encoder_train_mode** (default ``\"frozen\"``):
    - **frozen**: freeze **entire** text tower (BERT + projection). Only **image encoder** +
      **logit_scale** train. Text prompts stay distinct (CheXzero-style fixed language side).
      Training only BERT while still updating ``proj`` can still collapse both prompts to the
      same unit vector → softmax always 0.5 — freezing all of ``text_encoder`` avoids that.
    - **proj_only**: freeze BERT, train projection + image + logit_scale.
    - **full**: train all parameters (can collapse prompts; not recommended).

    **weight_decay**: L2 penalty for AdamW (default ``1e-4``).
    """
    if text_encoder_train_mode not in ("frozen", "proj_only", "full"):
        raise ValueError(f"text_encoder_train_mode must be frozen|proj_only|full, got {text_encoder_train_mode!r}")

    model.train()
    if text_encoder_train_mode == "frozen":
        for p in model.text_encoder.parameters():
            p.requires_grad = False
        model.text_encoder.eval()
        params = list(model.image_encoder.parameters()) + [model.logit_scale]
    elif text_encoder_train_mode == "proj_only":
        for p in model.text_encoder.bert.parameters():
            p.requires_grad = False
        model.text_encoder.bert.eval()
        model.text_encoder.proj.train()
        params = (
            list(model.image_encoder.parameters())
            + list(model.text_encoder.proj.parameters())
            + [model.logit_scale]
        )
    else:
        params = (
            list(model.image_encoder.parameters())
            + list(model.text_encoder.parameters())
            + [model.logit_scale]
        )

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    for ep in range(epochs):
        total = 0.0
        n = 0
        for x, y in tqdm(loader, desc=f"adult ep{ep+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            z_img = model.encode_image(x)
            if text_encoder_train_mode == "frozen":
                with torch.no_grad():
                    z_txt = model.encode_text_batch(input_ids, attention_mask)
            else:
                z_txt = model.encode_text_batch(input_ids, attention_mask)
            loss = clip_style_loss(z_img, z_txt, y, model.logit_scale)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)


def finetune_pediatric_clip(
    model: ImageTextModel,
    loader: DataLoader,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float = 1e-4,
) -> torch.Tensor:
    """
    Freeze BERT; fine-tune image tower (+ logit_scale) on pediatric data.
    Returns frozen text embeddings [2, D].

    **weight_decay**: AdamW weight decay (default ``1e-4``).
    """
    for p in model.text_encoder.parameters():
        p.requires_grad = False
    with torch.no_grad():
        text_emb = model.encode_text_batch(input_ids, attention_mask).detach()

    train_params = list(model.image_encoder.parameters()) + [model.logit_scale]
    opt = torch.optim.AdamW(train_params, lr=lr, weight_decay=weight_decay)

    model.train()
    for ep in range(epochs):
        for x, y in tqdm(loader, desc=f"pediatric ft ep{ep+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            z_img = model.encode_image(x)
            loss = clip_style_loss(z_img, text_emb, y, model.logit_scale)
            loss.backward()
            opt.step()
    return text_emb


@torch.no_grad()
def collect_image_embeddings(
    model: ImageTextModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    zs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        z = model.encode_image(x).cpu().numpy()
        zs.append(z)
        ys.append(y.numpy())
    return np.concatenate(zs, axis=0), np.concatenate(ys, axis=0)


def plot_tsne_before_after(
    emb_before: np.ndarray,
    emb_after: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    seed: int = 42,
) -> None:
    ts = TSNE(n_components=2, perplexity=min(30, len(labels) - 1), random_state=seed, init="pca")
    b2 = ts.fit_transform(emb_before)
    a2 = TSNE(n_components=2, perplexity=min(30, len(labels) - 1), random_state=seed, init="pca").fit_transform(
        emb_after
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, title in zip(axes, [b2, a2], ["Before pediatric fine-tune", "After pediatric fine-tune"]):
        for lab, name, c in [(0, "Normal", "tab:blue"), (1, "Pneumonia", "tab:orange")]:
            m = labels == lab
            ax.scatter(data[m, 0], data[m, 1], s=8, alpha=0.7, label=name, c=c)
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _tensor_to_display_rgb(x: torch.Tensor) -> np.ndarray:
    """Single ImageNet-normalized image tensor (C,H,W) -> (H,W,3) float in [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(3, 1, 1)
    t = x * std + mean
    t = t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return t


@torch.no_grad()
def plot_prediction_examples(
    model: ImageTextModel,
    loader: DataLoader,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    frozen_text_emb: Optional[torch.Tensor] = None,
    n_show: int = 12,
    out_path: Optional[Path] = None,
    title: str = "Pediatric test: ground truth vs prediction",
    show: bool = True,
) -> None:
    """
    Grid of pediatric (test) images with class names, softmax P(pneumonia), and color (green/red) if correct.
    Run after training so `model` and `frozen_text_emb` match your fine-tuned checkpoint.
    """
    model.eval()
    if frozen_text_emb is not None:
        text_emb = frozen_text_emb
    else:
        text_emb = model.encode_text_batch(input_ids, attention_mask)

    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        if sum(t.shape[0] for t in xs) >= n_show:
            break
    if not xs:
        raise RuntimeError("Loader is empty — check pediatric test manifest / paths.")
    xb = torch.cat(xs, dim=0)[:n_show]
    yb = torch.cat(ys, dim=0)[:n_show]
    xb_dev = xb.to(device)
    z = model.encode_image(xb_dev)
    logit_scale = model.logit_scale.exp().clamp(1, 100)
    logits = logit_scale * (z @ text_emb.t())
    probs = F.softmax(logits, dim=1).cpu().numpy()
    pred = logits.argmax(dim=1).cpu().numpy()
    y_true = yb.numpy().astype(np.int64)
    class_names = ["Normal", "Pneumonia"]

    ncols = 4
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols, 3.0 * nrows), squeeze=False)
    for i in range(n_show):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ax.imshow(_tensor_to_display_rgb(xb[i]))
        ax.axis("off")
        ok = pred[i] == y_true[i]
        p_pna = float(probs[i, 1])
        ax.set_title(
            f"GT: {class_names[y_true[i]]}\nPred: {class_names[pred[i]]}  P(pna)={p_pna:.3f}",
            fontsize=8,
            color=("green" if ok else "crimson"),
        )
    for j in range(n_show, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if out_path is not None:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def stratified_subset_indices(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(labels))
    out: List[int] = []
    for c in (0, 1):
        cls = idx[labels == c]
        rng.shuffle(cls)
        k = max(1, int(np.round(fraction * len(cls))))
        out.extend(cls[:k].tolist())
    return np.array(sorted(out))


def prepare_loaders_from_manifests(
    adult_csv: Path,
    ped_csv: Path,
    batch_size: int,
    image_size: int,
    remap: Optional[Dict[str, str]] = None,
    max_adult_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Returns adult_train, ped_train, ped_val, ped_test loaders."""
    adult = read_manifest_csv(adult_csv)
    ped = read_manifest_csv(ped_csv)
    if remap:
        adult = apply_remap_df(adult, remap)
        ped = apply_remap_df(ped, remap)
    adult, _ = filter_existing(adult)
    ped, _ = filter_existing(ped)

    adult_train = adult[adult["split"] == "train_val"]
    if max_adult_samples is not None and len(adult_train) > max_adult_samples:
        adult_train = adult_train.sample(max_adult_samples, random_state=42)

    ped_train = ped[ped["split"] == "train"]
    ped_val = ped[ped["split"] == "val"]
    ped_test = ped[ped["split"] == "test"]

    tfm = build_transform(image_size)
    loaders = []
    for df in (adult_train, ped_train, ped_val, ped_test):
        ds = ManifestImageDataset(df["path"].tolist(), df["label"].tolist(), tfm)
        shuffle = df is not ped_test
        loaders.append(
            DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)
        )
    return loaders[0], loaders[1], loaders[2], loaders[3]


def learning_curve_experiment(
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
) -> Dict[str, List[Dict[str, float]]]:
    """
    Run adult contrastive pretraining once, then for each pediatric subset fraction fine-tune
    a copy of that checkpoint (proposed) and fit a ResNet baseline trained only on pediatric data.
    """
    input_ids, attention_mask = load_prompt_tensors(prompt_path, device)

    model_adult = ImageTextModel(
        image_backbone=image_backbone,
        pretrained_image=True,
    ).to(device)
    train_adult_clip(
        model_adult,
        adult_train_loader,
        input_ids,
        attention_mask,
        device,
        adult_epochs,
        lr_adult,
    )
    adult_ckpt = copy.deepcopy(model_adult.state_dict())
    del model_adult

    labels = np.array(ped_train_dataset.labels)
    results_proposed: List[Dict[str, float]] = []
    results_baseline: List[Dict[str, float]] = []

    for frac in fractions:
        idx = stratified_subset_indices(labels, frac, seed)
        sub = Subset(ped_train_dataset, idx.tolist())
        ped_loader = DataLoader(sub, batch_size=batch_size, shuffle=True, num_workers=0)

        m = ImageTextModel(image_backbone=image_backbone, pretrained_image=True).to(device)
        m.load_state_dict(adult_ckpt)
        ft = finetune_pediatric_clip(m, ped_loader, input_ids, attention_mask, device, ped_epochs, lr_ped)
        ev = evaluate_clip_classifier(
            m, ped_test_loader, input_ids, attention_mask, device, frozen_text_emb=ft
        )
        results_proposed.append({"fraction": float(frac), "auc": ev.auc, "f1": ev.f1})
        del m

        torch.manual_seed(seed + int(1000 * frac))
        bl = train_baseline(
            ped_loader,
            None,
            device,
            baseline_epochs,
            lr_base,
            backbone="resnet18",
            pretrained=True,
        )
        evb = evaluate_baseline(bl, ped_test_loader, device)
        results_baseline.append({"fraction": float(frac), "auc": evb.auc, "f1": evb.f1})

    return {"proposed": results_proposed, "baseline": results_baseline}


def save_proposed_bundle(
    path: Path,
    model: ImageTextModel,
    frozen_text_emb: torch.Tensor,
    *,
    image_size: int,
    image_backbone: str,
    max_adult_samples: Optional[int] = None,
    batch_size: int = 16,
) -> None:
    """
    Save fine-tuned CLIP-style model + frozen text embeddings for later evaluation without retraining.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "frozen_text_emb": frozen_text_emb.detach().cpu(),
            "image_size": int(image_size),
            "image_backbone": str(image_backbone),
            "max_adult_samples": max_adult_samples,
            "batch_size": int(batch_size),
        },
        path,
    )


def load_proposed_bundle(
    path: Path,
    device: torch.device,
) -> Tuple[ImageTextModel, torch.Tensor, Dict[str, Any]]:
    """Load bundle written by :func:`save_proposed_bundle`."""
    path = Path(path)
    ckpt = torch.load(path, map_location=device)
    backbone = ckpt["image_backbone"]
    model = ImageTextModel(image_backbone=backbone, pretrained_image=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    text_emb = ckpt["frozen_text_emb"].to(device)
    meta: Dict[str, Any] = {
        "image_size": int(ckpt["image_size"]),
        "image_backbone": backbone,
        "max_adult_samples": ckpt.get("max_adult_samples"),
        "batch_size": int(ckpt.get("batch_size", 16)),
    }
    return model, text_emb, meta
