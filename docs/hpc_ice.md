# Running VizFold (ESMFold) on ICE

This guide covers running the ESMFold backend and attention export on an ICE cluster via SLURM.

## Prerequisites

- Access to ICE with GPU nodes
- Conda environment with PyTorch (CUDA), OpenFold, and `fair-esm` installed

## Environment

### Option A: Conda (recommended)

```bash
conda create -n openfold_env python=3.10
conda activate openfold_env
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install fair-esm[esmfold]
# From repo root:
pip install -e .
```

### Option B: Use existing OpenFold environment

If OpenFold is already installed, just add `fair-esm`:

```bash
conda activate openfold_env
pip install fair-esm[esmfold]
```

## Interactive GPU session (debugging)

Request an interactive GPU node:

```bash
salloc --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=02:00:00
```

Then:

```bash
module load cuda   # or your site's module
conda activate openfold_env
cd /path/to/vizfold-foundation
python -c "import torch; print(torch.cuda.is_available())"
python -c "import esm; print('fair-esm OK')"
```

## Submitting a batch job

1. Set environment variables (or edit `scripts/hpc/ice/run_esmf_ice.slurm`):

   - `FASTA` – path to input FASTA (single sequence)
   - `OUTDIR` – where to write outputs
   - `TRACE_MODE` – `none` or `attention`

2. Submit:

```bash
export FASTA=examples/monomer/fasta_dir_6KWC/6KWC.fasta
export OUTDIR=outputs/esmf_6KWC
sbatch scripts/hpc/ice/run_esmf_ice.slurm
```

3. Monitor:

```bash
squeue -u $USER
```

4. Logs and outputs:

- SLURM stdout/stderr: `outputs/logs/esmf_<jobid>.out` and `.err`
- Run outputs: under `OUTDIR`: `meta.json`, `structure/`, `attention_files/`, `logs.txt`

## Minimal smoke test (<5 minutes)

To verify the job runs without using much GPU time:

```bash
# Create a tiny FASTA (e.g. 30 residues)
echo -e ">tiny\nMKFLKFSLLTAVLLSVVFAFSSCGDDDDDD" > /tmp/tiny.fasta
export FASTA=/tmp/tiny.fasta
export OUTDIR=outputs/esmf_smoke
export TRACE_MODE=none
sbatch scripts/hpc/ice/run_esmf_ice.slurm
```

Then run with trace:

```bash
export TRACE_MODE=attention
export LAYERS="0,1,47"  # only save 3 layers
sbatch scripts/hpc/ice/run_esmf_ice.slurm
```

## Common issues

| Issue | What to check |
|-------|---------------|
| CUDA not found | Load correct `cuda` module; `nvidia-smi` on the node |
| `torch` not seeing GPU | Install PyTorch with CUDA: `conda install pytorch pytorch-cuda=...` |
| Missing `esm` package | `pip install fair-esm[esmfold]` |
| Missing OpenFold | OpenFold must be built with CUDA extensions |
| Disk quota | Use `OUTDIR` on scratch or project space, not home if limited |
| Job killed (OOM) | Increase `--mem` or use shorter sequence / `--trace_mode none` |

## Partition and resources

- Use your site's GPU partition name in the script if different (e.g. `#SBATCH --partition=gpu`).
- Adjust `--time`, `--mem`, and `--cpus-per-task` to match queue limits and job size.
