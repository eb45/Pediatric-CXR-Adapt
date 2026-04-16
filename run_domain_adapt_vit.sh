#!/bin/bash
# =============================================================================
# Run pediatric_cxr_domain_adaptation_vit.ipynb under SLURM (no Jupyter timeout)
#
# Usage (on cluster, from anywhere):
#   cd /hpc/group/naderilab/eb408/ECE685   # or your project root
#   mkdir -p logs
#   export CONDA_ENV=peft_proteomics_env    # optional: your env name
#   export NOTEBOOK=pediatric_cxr_domain_adaptation_vit.ipynb
#   sbatch slurm_run_domain_adapt_vit.sh
#
# Put the notebook in PROJECT_ROOT, or set NOTEBOOK to an absolute path.
# Edit #SBATCH lines below for your site (partition, account, time, GPU type).
# =============================================================================

#SBATCH --job-name=cxr-vit-da
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu          # uncomment and set for your cluster
##SBATCH --account=your_account     # uncomment if required

set -euo pipefail

# Project root (must contain cxr_engine.py, data/, and the notebook)
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
export NOTEBOOK="${NOTEBOOK:-pediatric_cxr_domain_adaptation_vit.ipynb}"

cd "${PROJECT_ROOT}"
mkdir -p logs

if [[ ! -f "${NOTEBOOK}" ]]; then
  echo "ERROR: notebook not found: ${PROJECT_ROOT}/${NOTEBOOK}"
  echo "Copy pediatric_cxr_domain_adaptation_vit.ipynb here or set NOTEBOOK=/full/path.ipynb"
  exit 1
fi
if [[ ! -f cxr_engine.py ]]; then
  echo "ERROR: cxr_engine.py not in ${PROJECT_ROOT} — fix PROJECT_ROOT / cd to ECE685 repo."
  exit 1
fi

echo "=== Job ${SLURM_JOB_ID:-local} ==="
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "Notebook: ${NOTEBOOK}"
date

# --- Optional: HPC modules (uncomment/edit for your site) ---
# module purge
# module load cuda/12.1
# module load cudnn

# --- Conda / venv: edit ONE of these patterns ---
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
# Optional path remap for manifests built on another filesystem:
# export CXR_PATH_REMAP='{"\/old\/prefix":"\/new\/prefix"}'

# Prefer running nbconvert as a module (works inside conda env)
OUT_NB="executed_${SLURM_JOB_ID:-local}_vit.ipynb"
echo "Executing -> ${OUT_NB}"

python -m jupyter nbconvert \
  --to notebook \
  --execute "${NOTEBOOK}" \
  --output "${OUT_NB}" \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3

echo "Done: ${PROJECT_ROOT}/${OUT_NB}"
date
