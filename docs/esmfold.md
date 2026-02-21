# ESMFold backend

The ESMFold backend runs [ESMFold](https://github.com/facebookresearch/esm) (via `fair-esm`) and writes **VizFold-compatible** trace archives: structure + optional attention and activation tensors with metadata.

## Install

fair-esm does not install all ESMFold dependencies by default. Use either:

**Option A – conda (recommended)**  
Use `environment-mac.yml` (Mac) or `environment.yml` (Linux):

```bash
conda env create -f environment-mac.yml   # or environment.yml on Linux
conda activate openfold-env
```

**Option B – pip (after PyTorch is installed)**  
From repo root:

```bash
pip install -r requirements-esmfold.txt
# Optional: pip install -e .  for vizfold package
```

`requirements-esmfold.txt` includes: dm-tree, einops, omegaconf, fair-esm.

## Run locally

**Structure only (fast):**

```bash
python run_pretrained_esmf.py \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out outputs/esmf_6KWC \
  --trace_mode none
```

**Structure + attention + activations:**

```bash
python run_pretrained_esmf.py \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out outputs/esmf_6KWC \
  --model esmfold_v1 \
  --device cuda \
  --trace_mode attention+activations \
  --layers all \
  --save_fp16
```

**Limit layers/heads (saves memory and disk):**

```bash
python run_pretrained_esmf.py \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out outputs/esmf_6KWC \
  --trace_mode attention \
  --layers 0,1,2,5 \
  --heads 0,1,2
```

## Output layout

After a run, `--out` contains:

```
outputs/esmf_6KWC/
  meta.json              # Run metadata (backend, model, shapes, seed, etc.)
  structure/
    predicted.pdb        # Predicted structure (PDB)
    predicted.pt          # Optional coordinate tensor
  trace/
    attention/
      layer_000.pt
      layer_001.pt
      ...
    activations/
      layer_000.pt
      ...
    index.json           # Maps layer/head to path, dtype, shape
  logs.txt               # Log lines from the run
```

## meta.json

Includes:

- `backend`, `model_name`, `date_time`, `device`, `dtype`
- `sequence_length`, `input_fasta_hash`, `input_fasta_path`
- `layer_count`, `head_count`, `trace_mode`, `tensor_format` (fp16/fp32)
- `shapes_recorded`: per-file shapes for attention and activations
- `seed`, `deterministic` (if set)
- `repo_commit` (if run from a git repo)

## Reproducibility

- `--seed 42` fixes the PyTorch RNG.
- `--deterministic` sets CuDNN deterministic mode (can be slower).

Both are recorded in `meta.json`.

## Long sequences

Attention storage is O(N²). For long proteins the script warns and suggests:

- `--trace_mode activations` (no attention), or
- `--layers 0,1,2` to save only a few layers.

## Running on ICE (SLURM)

See [hpc_ice.md](hpc_ice.md) for batch submission, environment setup, and a short smoke test.
