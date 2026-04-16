#!/bin/bash
# =============================================================================
# Run run_learning_curve.py under SLURM (ResNet or ViT proposed encoder).
#
# Choose architecture before submit:
#   export ARCH=resnet   # default: resnet18 proposed
#   export ARCH=vit      # vit_b_16 proposed
#
# Examples:
#   cd /hpc/group/naderilab/eb408/ECE685   # your project root
#   mkdir -p logs
#   export CONDA_ENV=your_env
#   export ARCH=vit
#   sbatch slurm_run_learning_curve.sh
#
# Or one line:
#   sbatch --export=ARCH=vit,CONDA_ENV=your_env slurm_run_learning_curve.sh
#
# Optional: pass extra CLI args to the Python script (space-separated):
#   export EXTRA_LC_ARGS="--max-adult-samples 4000 --lc-adult-epochs 3"
#   export OUT_DIR=outputs/lc_vit_run1
#
# Outputs go to outputs/ (or $OUT_DIR): learning_curve_*_{resnet|vit}.png/csv/json
# Edit #SBATCH lines for partition, account, time, GPU, memory.
# =============================================================================

#SBATCH --job-name=cxr-lc
#SBATCH --output=logs/slurm-lc-%j.out
#SBATCH --error=logs/slurm-lc-%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu
##SBATCH --account=your_account

set -euo pipefail

ARCH="${ARCH:-resnet}"
case "${ARCH}" in
  resnet|vit) ;;
  *)
    echo "ERROR: ARCH must be 'resnet' or 'vit', got: ${ARCH}"
    exit 1
    ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
cd "${PROJECT_ROOT}"
mkdir -p logs

if [[ ! -f run_learning_curve.py ]]; then
  echo "ERROR: run_learning_curve.py not found in ${PROJECT_ROOT}"
  exit 1
fi
if [[ ! -f cxr_engine.py ]]; then
  echo "ERROR: cxr_engine.py not in ${PROJECT_ROOT} — set PROJECT_ROOT to your ECE685 repo."
  exit 1
fi

echo "=== Job ${SLURM_JOB_ID:-local}  ARCH=${ARCH} ==="
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
date

# --- Optional HPC modules ---
# module purge
# module load cuda/12.1
# module load cudnn

if [[ -n "${CONDA_SH:-}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

if [[ -n "${CONDA_ENV:-}" ]]; then
  conda activate "${CONDA_ENV}"
fi

export PYTHONUNBUFFERED=1
# export CXR_PATH_REMAP='{"\/old\/prefix":"\/new\/prefix"}'

OUT_DIR_ARGS=()
if [[ -n "${OUT_DIR:-}" ]]; then
  OUT_DIR_ARGS=(--out-dir "${OUT_DIR}")
fi

# shellcheck disable=SC2086
# EXTRA_LC_ARGS is intentionally unquoted so users can pass multiple flags.
python run_learning_curve.py \
  --arch "${ARCH}" \
  "${OUT_DIR_ARGS[@]}" \
  ${EXTRA_LC_ARGS:-}

echo "Finished OK"
date
