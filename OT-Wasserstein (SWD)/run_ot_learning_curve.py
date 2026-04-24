#!/usr/bin/env python3
"""
Learning curve sweep for the Wasserstein OT model (proposed) vs ResNet baseline.

Mirrors run_learning_curve.py but calls ot_engine.ot_learning_curve_experiment.

Example:
    python run_ot_learning_curve.py --arch resnet --lambda-ot 0.5
    python run_ot_learning_curve.py --arch vit --lambda-ot 1.0 --n-projections 256
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from cxr_data import load_path_remap
from cxr_engine import prepare_loaders_from_manifests
from ot_engine import ot_learning_curve_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="OT learning curve: Wasserstein vs baseline")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--arch", choices=("resnet", "vit"), default="resnet",
                        help="resnet → resnet18, vit → vit_b_16")
    parser.add_argument("--image-backbone", type=str, default=None,
                        help="Override backbone name (e.g. resnet50)")
    parser.add_argument("--ped-epochs", type=int, default=5)
    parser.add_argument("--lr-adult", type=float, default=1e-4)
    parser.add_argument("--lr-ped", type=float, default=1e-4)
    parser.add_argument("--baseline-epochs", type=int, default=5)
    parser.add_argument("--lr-base", type=float, default=1e-4)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--lc-adult-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-adult-samples", type=int, default=None)
    # OT-specific
    parser.add_argument("--lambda-ot", type=float, default=0.5)
    parser.add_argument("--lambda-adv", type=float, default=0.1)
    parser.add_argument("--lambda-ped", type=float, default=1.0)
    parser.add_argument("--n-projections", type=int, default=256)
    args = parser.parse_args()

    project = (args.project_root or Path(__file__).resolve().parent).resolve()
    sys.path.insert(0, str(project))
    os.chdir(project)

    out_dir = (args.out_dir or (project / "outputs/ot")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve backbone
    if args.image_backbone:
        backbone = args.image_backbone
    elif args.arch == "resnet":
        backbone = "resnet18"
    else:
        backbone = "vit_b_16"

    file_tag = f"ot_{args.arch}"

    adult_csv = project / "data/processed/adult_manifest.csv"
    ped_csv   = project / "data/processed/pediatric_manifest.csv"
    prompt_pt = project / "data/processed/bert_prompt_tokens.pt"
    remap     = load_path_remap()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Backbone:", backbone, "| λ_ot:", args.lambda_ot,
          "| λ_adv:", args.lambda_adv, "| n_proj:", args.n_projections)

    adult_loader, ped_train_loader, _, ped_test_loader = prepare_loaders_from_manifests(
        adult_csv, ped_csv,
        batch_size=args.batch_size,
        image_size=args.image_size,
        remap=remap,
        max_adult_samples=args.max_adult_samples,
    )

    curve = ot_learning_curve_experiment(
        adult_loader,
        ped_train_loader.dataset,
        ped_test_loader,
        prompt_pt,
        device,
        fractions=args.fractions,
        adult_epochs=args.lc_adult_epochs,
        ped_epochs=args.ped_epochs,
        lr_adult=args.lr_adult,
        lr_ped=args.lr_ped,
        image_backbone=backbone,
        baseline_epochs=args.baseline_epochs,
        lr_base=args.lr_base,
        batch_size=args.batch_size,
        seed=args.seed,
        lambda_ot=args.lambda_ot,
        lambda_adv=args.lambda_adv,
        lambda_ped=args.lambda_ped,
        n_projections=args.n_projections,
    )

    df_p = pd.DataFrame(curve["proposed"])
    df_b = pd.DataFrame(curve["baseline"])

    print("\nOT Proposed:"); print(df_p.to_string(index=False))
    print("\nBaseline:");    print(df_b.to_string(index=False))

    df_p.to_csv(out_dir / f"lc_ot_proposed_{file_tag}.csv", index=False)
    df_b.to_csv(out_dir / f"lc_ot_baseline_{file_tag}.csv", index=False)

    (out_dir / f"lc_ot_full_{file_tag}.json").write_text(
        json.dumps({"backbone": backbone, "curve": curve}, indent=2)
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    for df, name, style in [(df_p, "OT (Wasserstein)", "o-"), (df_b, "Baseline", "s--")]:
        ax0.plot(df["fraction"], df["auc"], style, label=name)
        ax1.plot(df["fraction"], df["f1"],  style, label=name)
    for ax, ylabel, title in [
        (ax0, "AUC-ROC", "Pediatric test — AUC-ROC"),
        (ax1, "F1",      "Pediatric test — F1"),
    ]:
        ax.set_xlabel("Pediatric train fraction")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"lc_ot_auc_f1_{file_tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nDone. Outputs in:", out_dir)


if __name__ == "__main__":
    main()
