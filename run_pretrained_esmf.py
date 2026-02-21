#!/usr/bin/env python3
"""
ESMFold inference with VizFold-compatible trace export.

Produces an archive under --out with:
  meta.json, structure/, trace/attention/, trace/activations/, trace/index.json, logs.txt

Example:
  python run_pretrained_esmf.py \\
    --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \\
    --out outputs/esmf_6KWC \\
    --model esmfold_v1 \\
    --device cuda \\
    --trace_mode attention+activations \\
    --layers all \\
    --save_fp16
"""
import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ESMFold and export VizFold-compatible traces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        "--model",
        type=str,
        default="esmfold_v1",
        help="Model name (e.g. esmfold_v1).",
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
        default="attention+activations",
        choices=["none", "attention", "activations", "attention+activations"],
        help="What to extract: none (structure only), attention, activations, or both.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers to save: 'all' or '0,1,2' or '0:12'.",
    )
    parser.add_argument(
        "--heads",
        type=str,
        default="all",
        help="Heads to save: 'all' or '0,1,2'.",
    )
    parser.add_argument(
        "--save_fp16",
        action="store_true",
        help="Save trace tensors in fp16 to reduce size.",
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
    args = parser.parse_args()

    if not os.path.isfile(args.fasta):
        print(f"Error: FASTA not found: {args.fasta}", file=sys.stderr)
        return 1

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    try:
        from vizfold.backends.esmfold.inference import ESMFoldRunner
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    runner = ESMFoldRunner(
        model_name=args.model,
        device=args.device,
        seed=args.seed,
        deterministic=args.deterministic,
    )
    runner.run(
        fasta_path=args.fasta,
        out_dir=args.out,
        trace_mode=args.trace_mode,
        layers=args.layers,
        heads=args.heads,
        save_fp16=args.save_fp16,
    )
    print(f"Done. Outputs in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
