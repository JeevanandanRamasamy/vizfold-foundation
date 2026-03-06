"""
Trace schema and metadata helpers for ESMFold outputs.

Ensures VizFold-compatible archive format and publication-grade metadata.
"""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _git_head(repo_path: Optional[str] = None) -> Optional[str]:
    """Return the current git HEAD commit hash, or None if not in a repo."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        if repo_path:
            cmd = ["git", "-C", repo_path, "rev-parse", "HEAD"]
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _read_fasta_and_hash(path: str) -> Tuple[str, str]:
    """Read a FASTA file and return (sequence, sha256_hash_prefix)."""
    with open(path) as f:
        raw = f.read()
    lines = [l.strip() for l in raw.splitlines() if l.strip() and not l.startswith(">")]
    seq = "".join(lines)
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return seq, h


def build_meta(
    *,
    backend: str = "esmfold",
    model_name: str,
    out_dir: str,
    fasta_path: Optional[str] = None,
    device: str,
    dtype: str,
    sequence_length: int,
    fasta_hash: str,
    layer_count: int,
    head_count: int,
    trace_mode: str,
    trace_formats: List[str],
    shapes_recorded: Dict[str, Any],
    seed: Optional[int] = None,
    deterministic: bool = False,
    top_k: int = 50,
    triangle_residue_idx: Optional[int] = None,
    repo_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build meta.json content for a VizFold-compatible run.

    shapes_recorded: e.g. {"attention": {"msa_row_attn": [num_heads, N, N]}}
    """
    meta: Dict[str, Any] = {
        "backend": backend,
        "model_name": model_name,
        "date_time": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "dtype": dtype,
        "sequence_length": sequence_length,
        "input_fasta_hash": fasta_hash,
        "layer_count": layer_count,
        "head_count": head_count,
        "trace_mode": trace_mode,
        "trace_formats": trace_formats,
        "top_k": top_k,
        "shapes_recorded": shapes_recorded,
        "attention_types": [
            "msa_row_attn",
            # ESMFold does NOT have triangle attention — this is an
            # architectural difference from OpenFold/AlphaFold2. The ESMFold
            # trunk uses only the ESM language model transformer (MSA-style
            # self-attention) and a lightweight folding head.
        ],
        "notes": (
            "ESMFold lacks pair representation and triangle attention. "
            "Only MSA row attention is available for visualization."
        ),
    }
    if fasta_path:
        meta["input_fasta_path"] = fasta_path
    if seed is not None:
        meta["seed"] = seed
    if deterministic:
        meta["deterministic"] = True
    if triangle_residue_idx is not None:
        meta["triangle_residue_idx"] = triangle_residue_idx
        meta["triangle_attention_warning"] = (
            "triangle_residue_idx was set but ESMFold does not have triangle "
            "attention. This parameter is ignored."
        )
    commit = _git_head(repo_path)
    if commit:
        meta["repo_commit"] = commit
    return meta


def write_meta(meta: Dict[str, Any], out_dir: str) -> str:
    """Write meta.json to out_dir and return the path."""
    path = os.path.join(out_dir, "meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path
