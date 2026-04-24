#!/usr/bin/env python3
"""
Prepare CXR data for ViT + BERT training:

  • NIH ChestX-ray14 (adult): binary pneumonia vs. not pneumonia from Data_Entry_2017.csv
  • Pediatric (Kermany / Kaggle layout): train/test[/val], NORMAL vs PNEUMONIA folders

Keep this script in your project root (next to requirements.txt / data/).

--nih-dir / --chestxray14-dir defaults to <project>/data/nih
  (Data_Entry_2017.csv + images/ and/or images_001/ ... images_012/).
  If train_val_list.txt and test_list.txt exist, split is train_val vs test; else split train.
--data-dir defaults to ./data/chest_xray relative to cwd (override with /work/... if needed).
--output-dir defaults to <project>/data/processed.

Writes to --output-dir:
  - adult_manifest.csv (NIH), pediatric_manifest.csv (pediatric)
  - bert_prompt_tokens.pt, vit_image_config.json

Optional: --cache-images → tensor shards under output-dir/cache/

  python preprocess_data.py
  python preprocess_data.py --chestxray14-dir /work/.../nih --data-dir /work/.../chest_xray
  python preprocess_data.py --skip-nih          # pediatric only
  python preprocess_data.py --skip-pediatric    # NIH only
  python preprocess_data.py --download-pediatric
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

try:
    from torchvision import transforms as T
except ImportError:
    T = None  # type: ignore

try:
    from transformers import BertTokenizer
except ImportError as e:
    print("Install transformers: pip install transformers", file=sys.stderr)
    raise e

# Project directory 
PROJECT_ROOT = Path(__file__).resolve().parent

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BERT_MODEL = "bert-base-uncased"

DEFAULT_PROMPTS = {
    "pneumonia": "a chest x-ray showing pneumonia",
    "normal": "a normal chest x-ray with no acute findings",
}

KAGGLE_PED_DATASET = "paultimothymooney/chest-xray-pneumonia"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_under_cwd(p: Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p.resolve()
    return (Path.cwd() / p).resolve()


def download_pediatric_from_kaggle(data_root: Path) -> Path:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise RuntimeError(
            "Kaggle download requires: pip install kaggle\n"
            "Then put API credentials in ~/.kaggle/kaggle.json (Kaggle account → API → Create Token)."
        ) from e

    data_root = data_root.resolve()
    parent = data_root.parent
    parent.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        KAGGLE_PED_DATASET,
        path=str(parent),
        unzip=True,
        quiet=False,
    )

    candidates = [parent / "chest_xray", data_root]
    for c in candidates:
        if c.is_dir() and (c / "train").is_dir() and (c / "test").is_dir():
            return c.resolve()

    for child in parent.iterdir():
        if child.is_dir() and (child / "train").is_dir() and (child / "test").is_dir():
            if child.name == "chest_xray" or (child / "train" / "NORMAL").is_dir():
                return child.resolve()

    raise RuntimeError(
        f"Downloaded Kaggle dataset but could not find train/ and test/ under {parent}. "
        "Pass --data-dir to the folder that contains train/ and test/."
    )


def build_vit_transform(image_size: int, mean: Sequence[float], std: Sequence[float]):
    if T is not None:
        return T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=list(mean), std=list(std)),
            ]
        )
    return ViTTransformFallback(image_size, mean, std)


class ViTTransformFallback:
    def __init__(self, image_size: int, mean: Sequence[float], std: Sequence[float]) -> None:
        self.size = (image_size, image_size)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, im: Image.Image) -> torch.Tensor:
        im = im.resize(self.size, Image.Resampling.BICUBIC)
        arr = np.asarray(im).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        x = torch.from_numpy(arr).permute(2, 0, 1)
        x = (x - self.mean) / self.std
        return x


def load_image_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def parse_nih_labels(finding_labels: str) -> List[str]:
    if pd.isna(finding_labels) or not str(finding_labels).strip():
        return []
    return [x.strip() for x in str(finding_labels).split("|") if x.strip()]


def build_nih_image_index(nih_root: Path) -> Dict[str, Path]:
    """
    Map basename (Image Index) -> path. Supports:
      - single folder images/
      - multi-archive layout images_001 ... images_012 (official Box download)
    PNGs may live directly under those folders or in subfolders — scan recursively.
    """
    index: Dict[str, Path] = {}
    dirs: List[Path] = []
    single = nih_root / "images"
    if single.is_dir():
        dirs.append(single)
    for d in sorted(nih_root.glob("images_*")):
        if d.is_dir():
            dirs.append(d)
    exts = {".png", ".jpg", ".jpeg"}
    for d in dirs:
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            index[p.name] = p
    return index


def load_nih_official_split_basenames(nih_root: Path) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
    """Basenames from train_val_list.txt and test_list.txt (one filename per line)."""

    def read_names(fname: str) -> Optional[set[str]]:
        path = nih_root / fname
        if not path.is_file():
            return None
        names: set[str] = set()
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                names.add(Path(line).name)
        return names

    return read_names("train_val_list.txt"), read_names("test_list.txt")


def build_nih_manifest(
    nih_root: Path,
    frontal_only: bool,
    normal_no_finding_only: bool,
    use_official_splits: bool = True,
) -> pd.DataFrame:
    """
    Binary labels: 1 = Pneumonia in Finding Labels; 0 = no pneumonia
    (optionally restrict negatives to 'No Finding' only).

    If use_official_splits and both train_val_list.txt and test_list.txt exist under
    nih_root, sets split to 'train_val' or 'test' (NIH standard held-out test set).
    Otherwise split is 'train' for all rows.
    """
    csv_path = nih_root / "Data_Entry_2017.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"Image Index", "Finding Labels", "View Position"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Data_Entry_2017.csv missing columns: {missing}")

    if frontal_only:
        vp = df["View Position"].astype(str).str.upper()
        df = df[vp.isin(["PA", "AP"])].copy()

    img_index = build_nih_image_index(nih_root)
    if not img_index:
        raise RuntimeError(
            f"No PNG/JPEG files found under {nih_root} (searched images/ and images_*/ recursively). "
            "Confirm paths exist on this host, images finished unpacking, and you have read permission."
        )

    train_val_names, test_names = load_nih_official_split_basenames(nih_root)
    use_lists = (
        use_official_splits
        and train_val_names is not None
        and test_names is not None
        and len(train_val_names) > 0
        and len(test_names) > 0
    )
    if use_official_splits and not use_lists:
        print(
            "NIH: train_val_list.txt and/or test_list.txt missing or empty — "
            "using split='train' for all rows. Add both files in the dataset root for official train/test.",
            file=sys.stderr,
        )

    rows = []
    skipped_official = 0
    for _, row in df.iterrows():
        name = str(row["Image Index"]).strip()
        path = img_index.get(name)
        if path is None:
            continue
        if use_lists:
            if name in test_names:
                split_name = "test"
            elif name in train_val_names:
                split_name = "train_val"
            else:
                skipped_official += 1
                continue
        else:
            split_name = "train"

        labels = parse_nih_labels(row["Finding Labels"])
        has_pneumonia = "Pneumonia" in labels
        if has_pneumonia:
            y = 1
        else:
            if normal_no_finding_only:
                if labels != ["No Finding"] and labels != []:
                    continue
            y = 0
        rows.append(
            {
                "path": str(path.resolve()),
                "label": y,
                "split": split_name,
                "domain": "adult",
                "source": "nih_chestxray14",
            }
        )

    if use_lists and skipped_official:
        print(
            f"NIH: skipped {skipped_official:,} rows not in train_val or test lists (after filters).",
            file=sys.stderr,
        )

    out = pd.DataFrame(rows)
    if use_lists and not out.empty:
        print("NIH split counts:\n" + out["split"].value_counts().to_string(), file=sys.stderr)
    if out.empty:
        raise RuntimeError(
            "No NIH rows collected. Check Data_Entry_2017.csv matches files in images/ or images_*/ "
            "and relax --nih-normal-no-finding-only / frontal filters if needed."
        )
    return out


def collect_split(root: Path, split_name: str) -> List[dict]:
    split_dir = root / split_name
    if not split_dir.is_dir():
        return []
    class_map = {"PNEUMONIA": 1, "NORMAL": 0}
    rows = []
    for class_name, y in class_map.items():
        cdir = split_dir / class_name
        if not cdir.is_dir():
            continue
        for p in sorted(cdir.glob("*")):
            if p.suffix.lower() not in {".jpeg", ".jpg", ".png", ".bmp"}:
                continue
            rows.append(
                {
                    "path": str(p.resolve()),
                    "label": y,
                    "split": split_name,
                    "domain": "pediatric",
                    "source": "kermany",
                }
            )
    return rows


def stratified_train_val(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
    group_col: str = "split",
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    parts = []
    for split_val, g in df.groupby(group_col):
        g = g.reset_index(drop=True)
        if split_val == "train" and val_fraction > 0:
            labels = g["label"].values
            idx = np.arange(len(g))
            val_idx = []
            for c in [0, 1]:
                cls_idx = idx[labels == c]
                rng.shuffle(cls_idx)
                n_val = int(np.round(val_fraction * len(cls_idx)))
                val_idx.extend(cls_idx[:n_val].tolist())
            val_set = set(val_idx)
            g = g.copy()
            g["split"] = g.index.map(lambda i: "val" if i in val_set else "train")
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def tokenize_prompts(prompts: dict, bert_model: str, max_length: int) -> dict:
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    classes = ["pneumonia", "normal"]
    texts = [prompts[c] for c in classes]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "classes": classes,
        "texts": texts,
        "tokenizer_name": bert_model,
    }


def save_tensor_shard(
    paths: Sequence[str],
    labels: Sequence[int],
    transform,
    out_file: Path,
    chunk_size: int = 256,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    buf_paths: List[str] = []
    buf_labels: List[int] = []
    buf_tensors: List[torch.Tensor] = []

    def flush() -> None:
        nonlocal buf_paths, buf_labels, buf_tensors
        if not buf_paths:
            return
        x = torch.stack(buf_tensors, dim=0)
        chunks.append(
            {
                "paths": buf_paths.copy(),
                "labels": torch.tensor(buf_labels, dtype=torch.long),
                "x": x.contiguous(),
            }
        )
        buf_paths, buf_labels, buf_tensors = [], [], []

    for p_str, y in tqdm(list(zip(paths, labels)), desc=f"Caching {out_file.stem}"):
        p = Path(p_str)
        im = load_image_rgb(p)
        t = transform(im)
        buf_paths.append(str(p))
        buf_labels.append(int(y))
        buf_tensors.append(t)
        if len(buf_paths) >= chunk_size:
            flush()
    flush()

    torch.save(chunks, out_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ChestX-ray14 + pediatric CXR for ViT + BERT")
    parser.add_argument(
        "--nih-dir",
        "--chestxray14-dir",
        type=Path,
        default=PROJECT_ROOT / "data/nih",
        metavar="DIR",
        help="ChestX-ray14 root: Data_Entry_2017.csv and images/ (same as --chestxray14-dir; default: <project>/data/nih)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/chest_xray"),
        help="Pediatric root: train/, test/ [, val/] with NORMAL/ and PNEUMONIA/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed",
        help=f"Output directory (default: {PROJECT_ROOT / 'data/processed'})",
    )
    parser.add_argument("--skip-nih", action="store_true", help="Do not build NIH (ChestX-ray14) manifest")
    parser.add_argument("--skip-pediatric", action="store_true", help="Do not build pediatric manifest")
    parser.add_argument(
        "--no-frontal-filter",
        action="store_true",
        help="NIH: include all view positions (default: PA/AP frontal only)",
    )
    parser.add_argument(
        "--nih-normal-no-finding-only",
        action="store_true",
        help="NIH: negative class only when Finding Labels is No Finding (stricter)",
    )
    parser.add_argument(
        "--ignore-nih-official-splits",
        action="store_true",
        help="NIH: do not use train_val_list.txt / test_list.txt (split='train' for all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--bert-model", type=str, default=DEFAULT_BERT_MODEL)
    parser.add_argument("--max-text-length", type=int, default=64)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Pediatric: fraction of train for val if no val/ folder exists",
    )
    parser.add_argument(
        "--cache-images",
        action="store_true",
        help="Save ViT-normalized tensor shards under output-dir/cache/",
    )
    parser.add_argument("--cache-chunk-size", type=int, default=256)
    parser.add_argument(
        "--download-pediatric",
        action="store_true",
        help=f"Download Kaggle {KAGGLE_PED_DATASET} (needs kaggle + ~/.kaggle/kaggle.json)",
    )
    args = parser.parse_args()
    set_seed(args.seed)

    if args.skip_nih and args.skip_pediatric:
        print("Nothing to do: both --skip-nih and --skip-pediatric.", file=sys.stderr)
        sys.exit(1)

    out = resolve_under_cwd(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mean, std = IMAGENET_MEAN, IMAGENET_STD
    transform = build_vit_transform(args.image_size, mean, std)

    with open(out / "vit_image_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_size": args.image_size,
                "mean": list(mean),
                "std": list(std),
                "interpolation": "bicubic",
                "note": "ImageNet normalize (ViT)",
            },
            f,
            indent=2,
        )

    prompt_bundle = tokenize_prompts(DEFAULT_PROMPTS, args.bert_model, args.max_text_length)
    torch.save(prompt_bundle, out / "bert_prompt_tokens.pt")
    with open(out / "bert_prompts.json", "w", encoding="utf-8") as f:
        json.dump({"prompts": DEFAULT_PROMPTS, "bert_model": args.bert_model}, f, indent=2)

    nih_path = args.nih_dir if args.nih_dir.is_absolute() else resolve_under_cwd(args.nih_dir)
    did_nih = False

    if not args.skip_nih:
        if not nih_path.is_dir():
            print(
                f"Skipping NIH: not a directory: {nih_path}\n"
                "Place ChestX-ray14 here (Data_Entry_2017.csv + images/) or set --chestxray14-dir / --nih-dir.",
                file=sys.stderr,
            )
        else:
            try:
                adult = build_nih_manifest(
                    nih_path,
                    frontal_only=not args.no_frontal_filter,
                    normal_no_finding_only=args.nih_normal_no_finding_only,
                    use_official_splits=not args.ignore_nih_official_splits,
                )
                adult_path = out / "adult_manifest.csv"
                adult.to_csv(adult_path, index=False)
                print(f"Wrote {len(adult):,} rows -> {adult_path}")
                did_nih = True
                if args.cache_images:
                    save_tensor_shard(
                        adult["path"].tolist(),
                        adult["label"].tolist(),
                        transform,
                        out / "cache" / "adult_tensors.pt",
                        chunk_size=args.cache_chunk_size,
                    )
            except (FileNotFoundError, RuntimeError, ValueError) as e:
                print(f"Skipping NIH: {e}", file=sys.stderr)

    did_ped = False
    if not args.skip_pediatric:
        data_root = resolve_under_cwd(args.data_dir)

        if args.download_pediatric:
            print(f"Downloading pediatric CXR from Kaggle ({KAGGLE_PED_DATASET}) …", file=sys.stderr)
            data_root = download_pediatric_from_kaggle(data_root)
            print(f"Data root: {data_root}", file=sys.stderr)

        if not data_root.is_dir():
            print(
                f"Skipping pediatric: missing directory: {data_root}\n"
                "Pass --data-dir, use --download-pediatric, or --skip-pediatric.",
                file=sys.stderr,
            )
        else:
            train_r = collect_split(data_root, "train")
            val_r = collect_split(data_root, "val")
            test_r = collect_split(data_root, "test")
            rows = train_r + val_r + test_r
            if not rows:
                print(
                    f"Skipping pediatric: no images under {data_root} "
                    f"(need train/ and test/ with NORMAL/ and PNEUMONIA/).",
                    file=sys.stderr,
                )
            else:
                ped = pd.DataFrame(rows)
                if val_r:
                    print(
                        f"Using on-disk val/ ({len(val_r):,} images); not re-splitting train.",
                        file=sys.stderr,
                    )
                else:
                    ped = stratified_train_val(
                        ped, val_fraction=args.val_fraction, seed=args.seed, group_col="split"
                    )

                ped_path = out / "pediatric_manifest.csv"
                ped.to_csv(ped_path, index=False)
                print(f"Wrote {len(ped):,} rows -> {ped_path}")
                did_ped = True
                if args.cache_images:
                    cache_dir = out / "cache"
                    for split_name in ped["split"].unique():
                        sub = ped[ped["split"] == split_name]
                        save_tensor_shard(
                            sub["path"].tolist(),
                            sub["label"].tolist(),
                            transform,
                            cache_dir / f"pediatric_{split_name}_tensors.pt",
                            chunk_size=args.cache_chunk_size,
                        )

    if not did_nih and not did_ped:
        print(
            f"No manifests written. Fix --chestxray14-dir / --data-dir or use --download-pediatric. "
            f"Still wrote config under {out}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
