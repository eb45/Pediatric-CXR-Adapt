#!/bin/bash
# =============================================================================
# Run pediatric_cxr_domain_adaptation_ot.ipynb under SLURM
#
# Usage (from project root):
#   mkdir -p logs
#   export CONDA_ENV=your_env
#   sbatch slurm_run_ot.sh
#
# Optional overrides:
#   export NOTEBOOK=pediatric_cxr_domain_adaptation_ot.ipynb
#   export OT_MODE=sinkhorn          # 'swd' (default) or 'sinkhorn'
#   export LAMBDA_OT=0.5
#   export IMAGE_BACKBONE=vit_b_16
# =============================================================================

#SBATCH --job-name=cxr-ot-da
#SBATCH --output=logs/slurm-ot-%j.out
#SBATCH --error=logs/slurm-ot-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu
##SBATCH --account=your_account

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
export NOTEBOOK="${NOTEBOOK:-pediatric_cxr_domain_adaptation_ot.ipynb}"

cd "${PROJECT_ROOT}"
mkdir -p logs

if [[ ! -f "${NOTEBOOK}" ]]; then
  echo "ERROR: notebook not found: ${PROJECT_ROOT}/${NOTEBOOK}"
  exit 1
fi
if [[ ! -f ot_engine.py ]]; then
  echo "ERROR: ot_engine.py not in ${PROJECT_ROOT}"
  exit 1
fi

echo "=== Job ${SLURM_JOB_ID:-local} ==="
echo "Host: $(hostname)"
echo "PWD:  $(pwd)"
echo "Notebook: ${NOTEBOOK}"
date

# Optional conda setup
if [[ -n "${CONDA_SH:-}" ]]; then
  source "${CONDA_SH}"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

if [[ -n "${CONDA_ENV:-}" ]]; then
  conda activate "${CONDA_ENV}"
fi

export PYTHONUNBUFFERED=1
# export CXR_PATH_REMAP='{"\/old\/prefix":"\/new\/prefix"}'

OUT_NB="executed_${SLURM_JOB_ID:-local}_ot.ipynb"
echo "Executing → ${OUT_NB}"

python -m jupyter nbconvert \
  --to notebook \
  --execute "${NOTEBOOK}" \
  --output "${OUT_NB}" \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3

echo "Done: ${PROJECT_ROOT}/${OUT_NB}"
date
