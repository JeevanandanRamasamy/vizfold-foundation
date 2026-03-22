"""
Writes ESMFold traces into VizFold-compatible archive layout.

Layout:
  out_dir/
    meta.json
    structure/
      predicted.pdb
    attention_files/
      msa_row_attn_layerXX_headYY.txt    (standard VizFold text format)
    logs.txt
"""
import os
from typing import Any, Dict, Optional

from vizfold.backends.esmfold.schema import (
    build_meta,
    write_meta,
)


def write_structure(out_dir: str, pdb_str: Optional[str]) -> Dict[str, str]:
    """Save predicted PDB to structure/ subdirectory."""
    base = os.path.join(out_dir, "structure")
    os.makedirs(base, exist_ok=True)
    paths: Dict[str, str] = {}
    if pdb_str:
        pdb_path = os.path.join(base, "predicted.pdb")
        with open(pdb_path, "w") as f:
            f.write(pdb_str)
        paths["pdb"] = pdb_path
    return paths


def build_and_write_meta(
    out_dir: str,
    model_name: str,
    fasta_path: str,
    device: str,
    dtype: str,
    seq_len: int,
    fasta_hash: str,
    layer_count: int,
    head_count: int,
    trace_mode: str,
    shapes_recorded: Dict[str, Any],
    seed: Optional[int] = None,
    deterministic: bool = False,
    top_k: int = 50,
    triangle_residue_idx: Optional[int] = None,
) -> str:
    """Build and write meta.json for an ESMFold run."""
    meta = build_meta(
        backend="esmfold",
        model_name=model_name,
        out_dir=out_dir,
        fasta_path=fasta_path,
        device=device,
        dtype=dtype,
        sequence_length=seq_len,
        fasta_hash=fasta_hash,
        layer_count=layer_count,
        head_count=head_count,
        trace_mode=trace_mode,
        trace_formats=["txt"],  # VizFold standard text format
        shapes_recorded=shapes_recorded,
        seed=seed,
        deterministic=deterministic,
        top_k=top_k,
        triangle_residue_idx=triangle_residue_idx,
    )
    return write_meta(meta, out_dir)
