#!/bin/bash
#SBATCH --job-name=esmf_vizfold
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/logs/esmf_%j.out
#SBATCH --error=outputs/logs/esmf_%j.err
#
# ESMFold inference with VizFold trace export on ICE.
#
# This uses the OpenFold model-loading method (fair-esm), NOT the HuggingFace
# Transformers backend. Requires fair-esm and OpenFold to be installed.
#
# Usage:
#   export FASTA=/path/to/seq.fasta OUTDIR=outputs/esmf_run
#   sbatch run_esmf_ice.slurm
# Or override on the command line:
#   sbatch run_esmf_ice.slurm  (uses FASTA and OUTDIR from environment)
#
# For a quick smoke test (<5 min): use a short FASTA and --trace_mode none

set -e

# Defaults if not set
FASTA="${FASTA:-examples/monomer/fasta_dir_6KWC/6KWC.fasta}"
OUTDIR="${OUTDIR:-outputs/esmf_ice_${SLURM_JOB_ID}}"
TRACE_MODE="${TRACE_MODE:-attention}"
DEVICE="${DEVICE:-cuda}"
TOP_K="${TOP_K:-50}"
LAYERS="${LAYERS:-all}"

mkdir -p outputs/logs
mkdir -p "$OUTDIR"

# Uncomment and set for your ICE environment:
# module load cuda/11.8
# module load anaconda3
# source activate openfold_env

# Or conda in project:
# eval "$(conda shell.bash hook)"
# conda activate openfold_env

cd "${VIZFOLD_REPO:-.}"
echo "FASTA=$FASTA OUTDIR=$OUTDIR TRACE_MODE=$TRACE_MODE"
echo "Running ESMFold (OpenFold method)..."

python run_pretrained_esmf.py \
  --fasta "$FASTA" \
  --out "$OUTDIR" \
  --device "$DEVICE" \
  --trace_mode "$TRACE_MODE" \
  --layers "$LAYERS" \
  --top_k "$TOP_K"

echo "Done. Outputs in $OUTDIR"
