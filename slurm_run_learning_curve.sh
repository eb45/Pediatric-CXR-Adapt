#!/bin/bash

#SBATCH --job-name=cxr-lc
#SBATCH --output=logs/slurm-lc-%j.out
#SBATCH --error=logs/slurm-lc-%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

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

OUT_DIR_ARGS=()
if [[ -n "${OUT_DIR:-}" ]]; then
  OUT_DIR_ARGS=(--out-dir "${OUT_DIR}")
fi

EXTRA_LC_ARGS="${EXTRA_LC_ARGS:---pediatric-seeds 42 43 44}"
echo "EXTRA_LC_ARGS: ${EXTRA_LC_ARGS}"

python run_learning_curve.py \
  --arch "${ARCH}" \
  "${OUT_DIR_ARGS[@]}" \
  ${EXTRA_LC_ARGS}

echo "Finished OK"
date
