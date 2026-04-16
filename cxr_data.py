"""
Manifest loading and optional path remapping (HPC -> local) for NIH + pediatric CXR pipelines.
"""

from __future__ import annotations

import json
import os
import torch
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def load_path_remap() -> Dict[str, str]:
    raw = os.environ.get("CXR_PATH_REMAP", "").strip()
    if not raw:
        return {}
    try:
        m = json.loads(raw)
        return {str(k): str(v) for k, v in m.items()}
    except json.JSONDecodeError:
        # "old_prefix>new_prefix;old2>new2"
        out: Dict[str, str] = {}
        for part in raw.split(";"):
            part = part.strip()
            if ">" in part:
                a, b = part.split(">", 1)
                out[a.strip()] = b.strip()
        return out


def remap_path(p: str, mapping: Dict[str, str]) -> str:
    for old, new in sorted(mapping.items(), key=lambda x: -len(x[0])):
        if p.startswith(old):
            return new + p[len(old) :]
    return p


def read_manifest_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"path", "label", "split"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{path} missing columns: {miss}")
    return df


def apply_remap_df(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    if not mapping:
        return df
    out = df.copy()
    out["path"] = out["path"].map(lambda s: remap_path(str(s), mapping))
    return out


def filter_existing(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    keep = []
    n_miss = 0
    for _, row in df.iterrows():
        p = Path(row["path"])
        if p.is_file():
            keep.append(True)
        else:
            keep.append(False)
            n_miss += 1
    return df.loc[keep].reset_index(drop=True), n_miss


class ManifestImageDataset(Dataset):
    """Loads grayscale/RGB CXR images from a manifest (path, label)."""

    def __init__(
        self,
        paths: Sequence[str],
        labels: Sequence[int],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        p = Path(self.paths[i])
        with Image.open(p) as im:
            im = im.convert("RGB")
        x = self.transform(im)
        y = int(self.labels[i])
        return x, y


def plot_xray_grid(
    manifest_csv: Path,
    *,
    title: str = "",
    n_per_class: int = 4,
    remap: Optional[Dict[str, str]] = None,
    split: Optional[str] = None,
    out_path: Optional[Path] = None,
    seed: int = 0,
    class_names: Tuple[str, str] = ("Normal (label 0)", "Pneumonia (label 1)"),
    show: bool = True,
) -> None:
    """
    Show a 2-row grid: random samples per class from a manifest (grayscale).
    Optionally filter by split (e.g. 'train_val', 'test', 'train') when column exists.
    """
    import matplotlib.pyplot as plt

    df = read_manifest_csv(manifest_csv)
    if remap:
        df = apply_remap_df(df, remap)
    df, _n_miss = filter_existing(df)
    if df.empty:
        raise RuntimeError(f"No existing image paths in {manifest_csv} (check paths / CXR_PATH_REMAP).")
    if split is not None and "split" in df.columns:
        df = df[df["split"].astype(str) == split]
    if df.empty:
        raise RuntimeError(f"No rows after split={split!r} filter.")

    rng = np.random.RandomState(seed)
    fig, axes = plt.subplots(2, n_per_class, figsize=(2.2 * n_per_class, 5.0), squeeze=False)

    for row, label in enumerate((0, 1)):
        pool = df[df["label"].astype(int) == label]
        if pool.empty:
            for col in range(n_per_class):
                axes[row, col].axis("off")
            continue
        k = min(n_per_class, len(pool))
        pick = rng.choice(len(pool), size=k, replace=False)
        take = pool.iloc[pick].reset_index(drop=True)
        for col in range(n_per_class):
            ax = axes[row, col]
            if col < len(take):
                p = Path(str(take.iloc[col]["path"]))
                with Image.open(p) as im:
                    im = im.convert("L")
                    arr = np.asarray(im, dtype=np.float32)
                ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
                ax.set_title(Path(p).name[:32] + ("…" if len(Path(p).name) > 32 else ""), fontsize=7)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(class_names[row], fontsize=10)

    fig.suptitle(title or manifest_csv.name, fontsize=12)
    fig.tight_layout()
    if out_path is not None:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_adult_pediatric_layman_figure(
    adult_manifest_csv: Path,
    pediatric_manifest_csv: Path,
    *,
    remap: Optional[Dict[str, str]] = None,
    adult_split: Optional[str] = "train_val",
    pediatric_split: str = "train",
    out_path: Optional[Path] = None,
    seed: int = 42,
    show: bool = True,
) -> None:
    """
    One **2×2** figure for non-expert audiences:

    |                    | **Normal** (no pneumonia) | **Pneumonia** (lung infection) |
    |--------------------|---------------------------|--------------------------------|
    | **Adults** (NIH)   | sample                    | sample                         |
    | **Young children** | sample                  | sample                         |

    Short captions explain *domain adaptation* without clinical jargon.
    """
    import matplotlib.pyplot as plt

    def _prep(path: Path, split: Optional[str]) -> pd.DataFrame:
        df = read_manifest_csv(path)
        if remap:
            df = apply_remap_df(df, remap)
        df, _ = filter_existing(df)
        if split is not None and "split" in df.columns:
            df = df[df["split"].astype(str) == split]
        return df

    rng = np.random.RandomState(seed)
    adult_df = _prep(adult_manifest_csv, adult_split)
    ped_df = _prep(pediatric_manifest_csv, pediatric_split)
    if adult_df.empty or ped_df.empty:
        raise RuntimeError("Adult or pediatric manifest empty after filters (check split / paths).")

    fig, axes = plt.subplots(2, 2, figsize=(9, 9), squeeze=False)

    def _one_sample(df: pd.DataFrame, label: int) -> np.ndarray:
        pool = df[df["label"].astype(int) == label]
        if pool.empty:
            raise RuntimeError(f"No label={label} rows in manifest.")
        i = rng.randint(0, len(pool))
        p = Path(str(pool.iloc[i]["path"]))
        with Image.open(p) as im:
            im = im.convert("L")
            return np.asarray(im, dtype=np.float32)

    titles = [
        [
            "Adult chest X-ray — labeled normal\n(source: many hospital scans)",
            "Adult chest X-ray — labeled pneumonia\n(same source)",
        ],
        [
            "Child’s chest X-ray — labeled normal\n(target: smaller chest, different look)",
            "Child’s chest X-ray — labeled pneumonia\n(same disease, pediatric appearance)",
        ],
    ]

    for row, df in enumerate([adult_df, ped_df]):
        for col, lab in enumerate([0, 1]):
            ax = axes[row, col]
            arr = _one_sample(df, lab)
            ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            ax.set_title(titles[row][col], fontsize=10, pad=10)

    fig.suptitle(
        "Why “domain adaptation”? Adult vs pediatric chest X-rays (one random example per box)\n"
        "We pretrain on large adult data, then fine-tune on a small pediatric set.",
        fontsize=12,
        y=1.01,
    )

    caption = (
        "Lay summary: these are single examples, not diagnoses. "
        "Pediatric films often look different (patient size, positioning), so a model trained only on adults "
        "may not generalize—this project aligns adult-learned features to children."
    )
    fig.text(0.5, 0.01, caption, ha="center", fontsize=9, style="italic")

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    if out_path is not None:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
