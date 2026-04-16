#!/usr/bin/env python3
"""
Pediatric CXR: learning curve (proposed vs baseline).

Same logic as pediatric_cxr_learning_curve.ipynb: retrains adult contrastive
pretraining once, then sweeps pediatric train fractions for proposed (CLIP-style)
vs baseline (ResNet on pediatric only — see cxr_engine). Use --arch resnet or
--arch vit for the proposed image encoder (resnet18 vs vit_b_16). Writes
figures, CSVs, and prints tables.

Reduce LC_ADULT_EPOCHS or MAX_ADULT_SAMPLES for debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Non-interactive backend before pyplot import (works in SLURM / no display)
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch

from cxr_data import load_path_remap
from cxr_engine import learning_curve_experiment, prepare_loaders_from_manifests


def proposed_backbone_from_args(arch: str, image_backbone: str | None) -> str:
    """Resolve proposed ImageText backbone: explicit --image-backbone wins over --arch."""
    if image_backbone:
        return image_backbone
    if arch == "resnet":
        return "resnet18"
    if arch == "vit":
        return "vit_b_16"
    raise ValueError(arch)


def output_tag(arch: str, image_backbone: str | None, resolved: str) -> str:
    """Short string for output filenames (no path separators)."""
    if image_backbone:
        return resolved.replace("/", "_").replace(" ", "_")
    return arch


def main() -> None:
    parser = argparse.ArgumentParser(description="Learning curve: proposed vs baseline on pediatric test.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project directory (cxr_engine.py, data/processed/). Default: directory of this script.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNG/CSV/JSON. Default: <project>/outputs",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--arch",
        choices=("resnet", "vit"),
        default="resnet",
        help="Proposed model image encoder: resnet -> resnet18, vit -> vit_b_16.",
    )
    parser.add_argument(
        "--image-backbone",
        type=str,
        default=None,
        metavar="NAME",
        help="Override proposed backbone (e.g. resnet50). If set, --arch is ignored for the model.",
    )
    parser.add_argument("--ped-epochs", type=int, default=5)
    parser.add_argument("--lr-adult", type=float, default=1e-4)
    parser.add_argument("--lr-ped", type=float, default=1e-4)
    parser.add_argument("--baseline-epochs", type=int, default=5)
    parser.add_argument("--lr-base", type=float, default=1e-4)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 1.0],
        help="Pediatric train fractions to sweep.",
    )
    parser.add_argument("--lc-adult-epochs", type=int, default=5, help="Adult CLIP-style epochs inside the experiment.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-adult-samples", type=int, default=None, help="Cap adult manifest size (default: full).")
    args = parser.parse_args()

    project = (args.project_root or Path(__file__).resolve().parent).resolve()
    sys.path.insert(0, str(project))
    os.chdir(project)

    if not (project / "cxr_engine.py").is_file():
        print(f"ERROR: cxr_engine.py not found under project root: {project}", file=sys.stderr)
        sys.exit(1)

    out_dir = (args.out_dir or (project / "outputs")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    proposed_bb = proposed_backbone_from_args(args.arch, args.image_backbone)
    file_tag = output_tag(args.arch, args.image_backbone, proposed_bb)

    adult_csv = project / "data/processed/adult_manifest.csv"
    ped_csv = project / "data/processed/pediatric_manifest.csv"
    prompt_pt = project / "data/processed/bert_prompt_tokens.pt"
    remap = load_path_remap()

    print("Project:", project)
    print("ADULT_CSV:", adult_csv)
    print("PED_CSV:", ped_csv)
    print("PROMPT_PT:", prompt_pt)
    print("OUT_DIR:", out_dir)
    print("Proposed image backbone:", proposed_bb, f"(arch={args.arch})")
    print("Hyperparameters:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    adult_loader, ped_train_loader, _, ped_test_loader = prepare_loaders_from_manifests(
        adult_csv,
        ped_csv,
        batch_size=args.batch_size,
        image_size=args.image_size,
        remap=remap,
        max_adult_samples=args.max_adult_samples,
    )
    ped_train_ds = ped_train_loader.dataset

    curve = learning_curve_experiment(
        adult_loader,
        ped_train_ds,
        ped_test_loader,
        prompt_pt,
        device,
        args.fractions,
        args.lc_adult_epochs,
        args.ped_epochs,
        args.lr_adult,
        args.lr_ped,
        proposed_bb,
        baseline_epochs=args.baseline_epochs,
        lr_base=args.lr_base,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    df_p = pd.DataFrame(curve["proposed"])
    df_b = pd.DataFrame(curve["baseline"])

    print("Proposed:")
    print(df_p.to_string(index=False))
    print()
    print("Baseline:")
    print(df_b.to_string(index=False))
    print()

    csv_p = out_dir / f"learning_curve_proposed_{file_tag}.csv"
    csv_b = out_dir / f"learning_curve_baseline_{file_tag}.csv"
    df_p.to_csv(csv_p, index=False)
    df_b.to_csv(csv_b, index=False)
    print("Wrote:", csv_p)
    print("Wrote:", csv_b)

    json_path = out_dir / f"learning_curve_full_{file_tag}.json"
    payload = {
        "proposed_image_backbone": proposed_bb,
        "arch_flag": args.arch,
        "curve": curve,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print("Wrote:", json_path)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    for df, name, style in [(df_p, "Proposed", "o-"), (df_b, "Baseline", "s--")]:
        ax0.plot(df["fraction"], df["auc"], style, label=name)
        ax1.plot(df["fraction"], df["f1"], style, label=name)
    ax0.set_xlabel("Pediatric train fraction")
    ax0.set_ylabel("AUC-ROC")
    ax0.set_title("Pediatric test — AUC-ROC")
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    ax1.set_xlabel("Pediatric train fraction")
    ax1.set_ylabel("F1")
    ax1.set_title("Pediatric test — F1")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = out_dir / f"learning_curve_auc_f1_{file_tag}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", png_path)
    print("Done.")


if __name__ == "__main__":
    main()
