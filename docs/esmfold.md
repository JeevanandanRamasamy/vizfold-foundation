# ESMFold Backend

The ESMFold backend runs [ESMFold](https://github.com/facebookresearch/esm) through **OpenFold's model architecture**, enabling VizFold's full attention capture pipeline. This means attention maps are saved in the same text-file format as OpenFold, and existing visualization tools (PyMOL 3D, arc diagrams) work without modification.

## Architecture Note

> **ESMFold does NOT have triangle attention.** ESMFold uses only the ESM language model's transformer (MSA-style self-attention) and a lightweight folding head. There is no pair representation, no triangle multiplication, and no `TriangleAttentionStartingNode/EndingNode`. Only **MSA row attention** is available for visualization. For triangle attention, use `run_pretrained_openfold.py` with an AlphaFold2/OpenFold checkpoint.

## Install

### Prerequisites
- Working OpenFold installation (including CUDA build)
- Python 3.10+, PyTorch 2.0+

### ESMFold dependency

```bash
pip install -r requirements-esmfold.txt
```

This installs `fair-esm[esmfold]` (Meta's ESM library). The first run will automatically download the ESMFold checkpoint (~5.5 GB) and patch the IPA keys for VizFold compatibility.

### macOS (for development, CPU only)

```bash
conda env create -f environment-mac.yml
conda activate openfold-env
```

## Run Locally

### Structure only (fast)

```bash
python run_pretrained_esmf.py \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out outputs/esmf_6KWC \
  --trace_mode none
```

### Structure + MSA attention

```bash
python run_pretrained_esmf.py \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out outputs/esmf_6KWC \
  --device cuda \
  --trace_mode attention \
  --layers all \
  --top_k 50
```

### Limit layers (saves memory and disk)

```bash
python run_pretrained_esmf.py \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out outputs/esmf_6KWC \
  --trace_mode attention \
  --layers 0,1,2,47
```

## Output Layout

```
outputs/esmf_6KWC/
  meta.json                         # Run metadata (backend, model, FASTA hash, seed, etc.)
  structure/
    predicted.pdb                   # Predicted structure (PDB)
  attention_files/
    msa_row_attn_layer0.txt         # Standard VizFold top-K attention text files
    msa_row_attn_layer1.txt
    ...
    msa_row_attn_layer47.txt
  logs.txt                          # Log lines from the run
```

### Attention text file format

Each `.txt` file contains one section per head:
```
Layer 47, Head 0
12 34 0.052381
12 12 0.048129
...
Layer 47, Head 1
...
```

Each line has: `source_residue target_residue attention_score` (top-K by attention weight).

This is the **same format** produced by `run_pretrained_openfold.py --demo_attn`, so all existing visualization tools work directly:
- `visualize_attention_3d_demo_utils.py` (PyMOL 3D)
- `visualize_attention_arc_diagram_demo_utils.py` (arc diagrams)
- `visualize_attention_general_utils.py` (combined panels)

## meta.json

Includes:
- `backend` (`"esmfold"`), `model_name`, `date_time`, `device`, `dtype`
- `sequence_length`, `input_fasta_hash`, `input_fasta_path`
- `layer_count` (48), `head_count` (8), `trace_mode`, `top_k`
- `attention_types`: `["msa_row_attn"]` (no triangle attention)
- `seed`, `deterministic` (if set)
- `repo_commit` (if run from a git repo)

## Reproducibility

- `--seed 42` fixes the PyTorch RNG.
- `--deterministic` sets CuDNN deterministic mode (can be slower).

Both are recorded in `meta.json`.

## Comparison: ESMFold vs OpenFold Attention

| Feature | OpenFold (`run_pretrained_openfold.py`) | ESMFold (`run_pretrained_esmf.py`) |
|---|---|---|
| MSA row attention | ✅ 48 layers × 8 heads | ✅ 48 layers × 8 heads |
| Triangle start attention | ✅ with `--triangle_residue_idx` | ❌ Not available |
| Triangle end attention | ✅ | ❌ Not available |
| MSA column attention | ✅ | ❌ Not available |
| Output format | Text files (`.txt`) | Text files (`.txt`) — same format |
| Visualization tools | All work | All work (MSA row only) |
| Requires MSA/alignments | Yes | No (single sequence) |

## Running on ICE (SLURM)

See [hpc_ice.md](hpc_ice.md) for batch submission, environment setup, and a short smoke test.
