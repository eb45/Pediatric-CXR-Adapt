"""
Evaluation figures: ROC/PR, raw + row-normalized confusion matrices, score histograms + boxplots,
threshold sweeps (F1, Youden J), calibration, per-class metrics (sens/spec/PPV/NPV) bar chart,
bootstrap AUC CI, failure-mode grid, Grad-CAM (ResNet), ROC overlay and metric bars for two models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc as sk_auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from cxr_model import ImageTextModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, Tuple[float, float]]:
    """Return sklearn ROC-AUC and (low, high) bootstrap percentile interval."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs: List[float] = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if yt.min() == yt.max():
            continue
        aucs.append(float(roc_auc_score(yt, ys)))
    if not aucs:
        return float("nan"), (float("nan"), float("nan"))
    point = float(roc_auc_score(y_true, y_score))
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, (float(lo), float(hi))


def run_evaluation_figures(
    labels: np.ndarray,
    probs: np.ndarray,
    out_dir: Path,
    *,
    prefix: str = "pediatric_test",
    bootstrap_n: int = 2000,
    calibration_bins: int = 10,
    thresholds: int = 401,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Save ROC, PR, confusion matrix, score histograms, threshold sweep, calibration plot,
    bootstrap AUC text file. Returns dict of scalar metrics.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y = labels.astype(np.int64).ravel()
    p = probs.astype(np.float64).ravel()

    metrics: Dict[str, float] = {}
    metrics["auc"] = float(roc_auc_score(y, p))
    pred05 = (p >= 0.5).astype(np.int64)
    metrics["f1_at_0.5"] = float(f1_score(y, pred05, zero_division=0))

    # --- ROC ---
    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = sk_auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — pediatric test")
    ax.legend(loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_roc.png", dpi=150)
    plt.close(fig)

    # --- Precision–recall ---
    prec, rec, _ = precision_recall_curve(y, p)
    pr_auc = sk_auc(rec, prec)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, lw=2, label=f"PR AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall — pediatric test")
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_pr.png", dpi=150)
    plt.close(fig)

    # --- Confusion matrix @ 0.5 ---
    cm = confusion_matrix(y, pred05, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred N", "Pred P"])
    ax.set_yticklabels(["True N", "True P"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black", fontsize=14)
    ax.set_title("Confusion matrix (threshold = 0.5)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- Score histograms ---
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 40)
    ax.hist(p[y == 0], bins=bins, alpha=0.65, label="True normal", color="tab:blue", density=True)
    ax.hist(p[y == 1], bins=bins, alpha=0.65, label="True pneumonia", color="tab:orange", density=True)
    ax.set_xlabel("Predicted P(pneumonia)")
    ax.set_ylabel("Density")
    ax.set_title("Score distributions by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_score_histograms.png", dpi=150)
    plt.close(fig)

    # --- Threshold sweep: F1 ---
    ts = np.linspace(0, 1, thresholds)
    f1s = [f1_score(y, (p >= t).astype(np.int64), zero_division=0) for t in ts]
    best_t = float(ts[int(np.argmax(f1s))])
    metrics["best_f1_threshold"] = best_t
    metrics["best_f1"] = float(np.max(f1s))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, f1s, lw=2, color="darkgreen")
    ax.axvline(0.5, color="gray", ls="--", label="t = 0.5")
    ax.axvline(best_t, color="crimson", ls=":", label=f"best t = {best_t:.3f}")
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("F1")
    ax.set_title("F1 vs threshold (pediatric test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_f1_vs_threshold.png", dpi=150)
    plt.close(fig)

    # --- Calibration (reliability) ---
    try:
        prob_true, prob_pred = calibration_curve(y, p, n_bins=calibration_bins, strategy="uniform")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        ax.plot(prob_pred, prob_true, "s-", label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Calibration ({calibration_bins} bins)")
        ax.legend()
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_calibration.png", dpi=150)
        plt.close(fig)
    except ValueError:
        pass

    # --- Normalized confusion matrix (row = true class proportions) ---
    # `cm` already computed above for raw counts
    row_sum = cm.sum(axis=1, keepdims=True).astype(np.float64)
    row_sum[row_sum == 0] = 1.0
    cm_norm = cm.astype(np.float64) / row_sum
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred N", "Pred P"])
    ax.set_yticklabels(["True N", "True P"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black", fontsize=13)
    ax.set_title("Confusion matrix normalized by row (true-class recall mix)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_confusion_matrix_normalized.png", dpi=150)
    plt.close(fig)

    # --- Per-class clinical-style metrics @ 0.5 ---
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    metrics["sensitivity"] = float(sens)
    metrics["specificity"] = float(spec)
    metrics["ppv"] = float(ppv)
    metrics["npv"] = float(npv)
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["Sensitivity\n(recall PNA)", "Specificity\n(recall N)", "PPV", "NPV"]
    vals = [sens, spec, ppv, npv]
    colors = ["tab:orange", "tab:blue", "tab:green", "tab:purple"]
    ax.bar(names, vals, color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Value")
    ax.set_title("Per-metrics @ threshold = 0.5 (pediatric test)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_per_class_metrics_bar.png", dpi=150)
    plt.close(fig)

    # --- Boxplot of scores by true label ---
    fig, ax = plt.subplots(figsize=(5, 4))
    data = [p[y == 0], p[y == 1]]
    if all(len(d) > 0 for d in data):
        ax.boxplot(data, labels=["True normal", "True pneumonia"], patch_artist=True)
        ax.set_ylabel("Predicted P(pneumonia)")
        ax.set_title("Score distribution by true class (boxplot)")
    else:
        ax.text(0.5, 0.5, "Need both classes for boxplot", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_score_boxplot_by_class.png", dpi=150)
    plt.close(fig)

    # --- Youden's J vs threshold (TPR − FPR along ROC) ---
    # sklearn `roc_curve` threshold length can differ by 1 across versions; align before plotting.
    fpr_u, tpr_u, th_roc = roc_curve(y, p)
    j_scores = tpr_u[1:] - fpr_u[1:]
    n = min(len(th_roc), len(j_scores))
    fig, ax = plt.subplots(figsize=(6, 4))
    if n > 0:
        ax.plot(th_roc[:n], j_scores[:n], lw=2, color="darkviolet")
    ax.axvline(0.5, color="gray", ls="--", alpha=0.7)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Youden's J (TPR − FPR)")
    ax.set_title("Youden's J vs threshold")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_youden_vs_threshold.png", dpi=150)
    plt.close(fig)

    # --- Bootstrap AUC ---
    point, (lo, hi) = bootstrap_auc_ci(y, p, n_boot=bootstrap_n, seed=seed)
    metrics["auc_bootstrap_lo"] = lo
    metrics["auc_bootstrap_hi"] = hi
    report = (
        f"Pediatric test evaluation\n"
        f"N = {len(y)}\n"
        f"ROC-AUC (sklearn) = {point:.4f}\n"
        f"ROC-AUC 95% bootstrap CI [{bootstrap_n} samples] = [{lo:.4f}, {hi:.4f}]\n"
        f"F1 @ 0.5 = {metrics['f1_at_0.5']:.4f}\n"
        f"Best F1 = {metrics['best_f1']:.4f} at threshold = {metrics['best_f1_threshold']:.4f}\n"
        f"Sensitivity @ 0.5 = {metrics['sensitivity']:.4f}\n"
        f"Specificity @ 0.5 = {metrics['specificity']:.4f}\n"
        f"PPV @ 0.5 = {metrics['ppv']:.4f}\n"
        f"NPV @ 0.5 = {metrics['npv']:.4f}\n"
    )
    (out_dir / f"{prefix}_metrics.txt").write_text(report, encoding="utf-8")

    return metrics


def pick_failure_indices(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, List[int]]:
    """Return example indices for TP, TN, FP, FN (first occurrences capped per category)."""
    pred = (probs >= threshold).astype(np.int64)
    y = labels.astype(np.int64)
    tp = np.where((y == 1) & (pred == 1))[0]
    tn = np.where((y == 0) & (pred == 0))[0]
    fp = np.where((y == 0) & (pred == 1))[0]
    fn = np.where((y == 1) & (pred == 0))[0]
    return {"tp": tp.tolist(), "tn": tn.tolist(), "fp": fp.tolist(), "fn": fn.tolist()}


def plot_failure_mode_grid(
    paths: Sequence[str],
    labels: np.ndarray,
    probs: np.ndarray,
    indices: Sequence[int],
    titles: Sequence[str],
    out_path: Path,
    ncols: int = 4,
) -> None:
    """Show raw PNG paths (grayscale) with GT / P(pna) / category."""
    n = len(indices)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 3.0 * nrows), squeeze=False)
    for k, idx in enumerate(indices):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        with Image.open(paths[idx]) as im:
            arr = np.asarray(im.convert("L"), dtype=np.float32)
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        ax.set_title(
            f"{titles[k]}\nGT={int(labels[idx])} P(pna)={probs[idx]:.3f}",
            fontsize=8,
        )
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")
    fig.suptitle("Failure analysis (sample indices)", fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def gradcam_resnet_last_layer(
    model: ImageTextModel,
    x: torch.Tensor,
    text_emb: torch.Tensor,
    target_class: int = 1,
) -> np.ndarray:
    """
    Grad-CAM heatmap [H', W'] for ResNet trunk (layer4). x: [1,3,H,W] on same device as model.
    """
    trunk = model.image_encoder.trunk
    if not hasattr(trunk, "layer4"):
        raise ValueError("Grad-CAM requires a ResNet backbone (layer4).")

    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def fwd_hook(_m, _inp, out):
        activations.append(out)

    def bwd_hook(_m, _gi, go):
        gradients.append(go[0])

    h1 = trunk.layer4.register_forward_hook(fwd_hook)
    h2 = trunk.layer4.register_full_backward_hook(bwd_hook)
    try:
        x_in = x.clone().detach().requires_grad_(True)
        z = model.encode_image(x_in)
        logit_scale = model.logit_scale.exp().clamp(1, 100)
        logits = logit_scale * (z @ text_emb.t())
        loss = logits[0, target_class]
        loss.backward()
    finally:
        h1.remove()
        h2.remove()

    a = activations[0][0]
    g = gradients[0][0]
    w = g.mean(dim=(1, 2), keepdim=True)
    cam = (w * a).sum(dim=0)
    cam = F.relu(cam)
    cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def overlay_heatmap_on_gray(
    gray: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """gray: [H,W] float in [0,255] or [0,1]; cam: [h,w] in [0,1], resized to gray if needed."""
    gh, gw = gray.shape
    ch, cw = cam.shape
    if (ch, cw) != (gh, gw):
        cam_img = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8))
        cam = np.asarray(cam_img.resize((gw, gh), Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
    g = gray.astype(np.float32)
    if g.max() > 1.5:
        g = g / 255.0
    g = np.clip(g, 0, 1)
    heat = plt.cm.jet(cam)[:, :, :3]
    rgb = (1 - alpha) * np.stack([g, g, g], axis=-1) + alpha * heat
    return np.clip(rgb, 0, 1)


def plot_gradcam_panel(
    model: ImageTextModel,
    dataset,  # ManifestImageDataset
    indices: Sequence[int],
    text_emb: torch.Tensor,
    device: torch.device,
    out_path: Path,
) -> None:
    """Save one row per index: input (gray) | Grad-CAM overlay. ResNet backbones only."""
    if not hasattr(model.image_encoder.trunk, "layer4"):
        raise ValueError("Grad-CAM panel requires ResNet backbone.")

    n = len(indices)
    fig, axes = plt.subplots(n, 2, figsize=(6, 2.8 * n), squeeze=False)
    model.eval()
    text_emb = text_emb.detach()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)

    for k, idx in enumerate(indices):
        y_true = int(dataset.labels[idx])
        x = dataset[idx][0].unsqueeze(0).to(device)
        H, W = x.shape[-2:]
        with torch.enable_grad():
            cam = gradcam_resnet_last_layer(model, x, text_emb, target_class=1)
        cam_t = torch.from_numpy(cam).float().to(device).unsqueeze(0).unsqueeze(0)
        cam_up = F.interpolate(cam_t, size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

        x0 = x[0].detach().cpu()
        img = (x0 * std.cpu().view(3, 1, 1) + mean.cpu().view(3, 1, 1)).clamp(0, 1)
        gray = img.mean(dim=0).numpy()
        gray_u8 = (gray * 255.0).astype(np.float32)
        rgb = overlay_heatmap_on_gray(gray_u8, cam_up, alpha=0.5)

        ax0 = axes[k, 0]
        ax0.imshow(gray, cmap="gray", vmin=0, vmax=1)
        ax0.axis("off")
        ax0.set_title(f"idx={idx} GT={y_true} (normalized input)", fontsize=8)

        ax1 = axes[k, 1]
        ax1.imshow(rgb)
        ax1.axis("off")
        ax1.set_title("Grad-CAM (pneumonia logit)", fontsize=8)

    fig.suptitle("Grad-CAM — ResNet layer4", fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_comparison(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    label_a: str,
    label_b: str,
    out_path: Path,
    *,
    title: str = "ROC comparison (same test set)",
) -> None:
    """Overlay two ROC curves for the same binary labels (e.g. proposed vs baseline)."""
    y = labels.astype(np.int64).ravel()
    fpr_a, tpr_a, _ = roc_curve(y, probs_a.ravel())
    fpr_b, tpr_b, _ = roc_curve(y, probs_b.ravel())
    auc_a = sk_auc(fpr_a, tpr_a)
    auc_b = sk_auc(fpr_b, tpr_b)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr_a, tpr_a, lw=2, label=f"{label_a} (AUC={auc_a:.4f})")
    ax.plot(fpr_b, tpr_b, lw=2, label=f"{label_b} (AUC={auc_b:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics_comparison_bars(
    models: Sequence[str],
    aucs: Sequence[float],
    f1s: Sequence[float],
    out_path: Path,
) -> None:
    """Side-by-side bar chart for AUC and F1 across models."""
    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 4))
    ax.bar(x - w / 2, aucs, w, label="AUC", color="steelblue", edgecolor="black")
    ax.bar(x + w / 2, f1s, w, label="F1 @ 0.5", color="coral", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison (pediatric test)")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
