#!/usr/bin/env python3
"""
ESMFold inference with VizFold-compatible trace export.

Runs ESMFold through OpenFold's model architecture, leveraging the existing
ATTENTION_METADATA attention capture pipeline. Produces attention maps in
the standard VizFold text-file format that visualization tools consume.

NOTE: ESMFold has no triangle attention (no pair representation). Only MSA
row attention is available. For triangle attention, use run_pretrained_openfold.py.

Example (structure only):
  python run_pretrained_esmf.py \\
    --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \\
    --out outputs/esmf_6KWC \\
    --trace_mode none

Example (structure + attention):
  python run_pretrained_esmf.py \\
    --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \\
    --out outputs/esmf_6KWC \\
    --device cuda \\
    --trace_mode attention \\
    --layers all \\
    --top_k 50
"""
import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ESMFold and export VizFold-compatible traces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--fasta",
        type=str,
        required=True,
        help="Path to FASTA file (single sequence).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory (will be created).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, cuda:0, cpu. Default: cuda if available else cpu.",
    )
    parser.add_argument(
        "--trace_mode",
        type=str,
        default="attention",
        choices=["none", "attention"],
        help=(
            "What to extract: none (structure only) or attention "
            "(structure + attention maps in VizFold text format). "
            "Note: ESMFold only supports MSA row attention, not triangle attention."
        ),
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers to save: 'all' or '0,1,2' or '0:12'.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Number of top attention values to save per head (default: 50).",
    )
    parser.add_argument(
        "--triangle_residue_idx",
        type=int,
        default=None,
        help=(
            "Residue index for triangle attention visualization. "
            "WARNING: ESMFold does not have triangle attention — this is ignored. "
            "Use run_pretrained_openfold.py for triangle attention."
        ),
    )
    parser.add_argument(
        "--attn_map_dir",
        type=str,
        default=None,
        help="Directory for attention text files. Default: <out>/attention_files.",
    )
    parser.add_argument(
        "--num_recycles_save",
        type=int,
        default=None,
        help="Number of recycling iterations to save attention for.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic CuDNN (may reduce speed).",
    )
    parser.add_argument(
        "--skip_relaxation",
        action="store_true",
        default=True,
        help="Skip Amber relaxation (default: True, ESMFold output is already relaxed-quality).",
    )
    args = parser.parse_args()

    # Validate FASTA
    if not os.path.isfile(args.fasta):
        print(f"Error: FASTA not found: {args.fasta}", file=sys.stderr)
        return 1

    # Auto-detect device
    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    # Import and run
    try:
        from vizfold.backends.esmfold.inference import ESMFoldRunner
    except ImportError as e:
        print(f"Error importing ESMFold backend: {e}", file=sys.stderr)
        print("Make sure fair-esm is installed: pip install fair-esm[esmfold]", file=sys.stderr)
        return 1

    runner = ESMFoldRunner(
        device=args.device,
        seed=args.seed,
        deterministic=args.deterministic,
    )
    result = runner.run(
        fasta_path=args.fasta,
        out_dir=args.out,
        trace_mode=args.trace_mode,
        layers=args.layers,
        top_k=args.top_k,
        triangle_residue_idx=args.triangle_residue_idx,
        attn_map_dir=args.attn_map_dir,
        num_recycles_save=args.num_recycles_save,
    )
    print(f"Done. Outputs in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
