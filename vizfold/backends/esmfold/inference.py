"""
ESMFold inference: load model, run forward, optionally extract traces.

Uses fair-esm (esm.pretrained.esmfold_v1) when available.
"""
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch

from vizfold.backends.esmfold.hooks import ESMFoldTraceCollector
from vizfold.backends.esmfold.schema import _read_fasta_and_hash
from vizfold.backends.esmfold.trace_adapter import (
    build_and_write_meta,
    write_structure,
    write_traces,
    write_trace_summary,
)

# Optional: fair-esm
try:
    import esm
    from esm.pretrained import esmfold_v1
    HAS_ESM = True
except ImportError:
    HAS_ESM = False
    esm = None
    esmfold_v1 = None


LONG_SEQ_WARN_THRESHOLD = 400  # N^2 attention warning


def _parse_layers_arg(layers_arg: Optional[str]) -> Optional[List[int]]:
    if not layers_arg or layers_arg.lower() == "all":
        return None
    indices = []
    for part in layers_arg.split(","):
        part = part.strip()
        if ":" in part:
            a, b = part.split(":", 1)
            indices.extend(range(int(a), int(b)))
        else:
            indices.append(int(part))
    return sorted(set(indices))


def _parse_heads_arg(heads_arg: Optional[str]) -> Optional[List[int]]:
    if not heads_arg or heads_arg.lower() == "all":
        return None
    return [int(x.strip()) for x in heads_arg.split(",")]


def read_fasta(fasta_path: str) -> Tuple[str, str]:
    """Return (sequence, id)."""
    seq, _ = _read_fasta_and_hash(fasta_path)
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                return seq, line[1:].strip().split()[0]
    return seq, "seq"


class ESMFoldRunner:
    """
    Runs ESMFold inference and writes VizFold-compatible output.

    Not a full BackendBase implementation; used by run_pretrained_esmf.py.
    """

    def __init__(
        self,
        model_name: str = "esmfold_v1",
        device: str = "cpu",
        dtype: Optional[str] = None,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ):
        if not HAS_ESM:
            raise RuntimeError(
                "ESMFold backend requires fair-esm. Install with: pip install fair-esm"
            )
        self.model_name = model_name
        self.device = device
        self.dtype = dtype or "float32"
        self.seed = seed
        self.deterministic = deterministic
        self._model = None
        self._alphabet = None

    def load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self.seed is not None:
            torch.manual_seed(self.seed)
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            warnings.warn("Deterministic mode may reduce speed.", UserWarning)

        model, alphabet = esmfold_v1()
        self._alphabet = alphabet
        model = model.eval()
        dtype = torch.float16 if self.dtype == "float16" else torch.float32
        model = model.to(device=self.device, dtype=dtype)
        self._model = model
        return model

    def run(
        self,
        fasta_path: str,
        out_dir: str,
        trace_mode: str = "attention+activations",
        layers: Optional[str] = None,
        heads: Optional[str] = None,
        save_fp16: bool = False,
        log_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run inference and write structure + optional traces.

        trace_mode: "attention" | "activations" | "attention+activations" | "none"
        layers: "all" or "0,1,2" or "0:12"
        heads: "all" or "0,1,2"
        """
        os.makedirs(out_dir, exist_ok=True)
        if log_path is None:
            log_path = os.path.join(out_dir, "logs.txt")

        def log(msg: str) -> None:
            with open(log_path, "a") as f:
                f.write(msg + "\n")
            print(msg)

        seq, seq_id = read_fasta(fasta_path)
        seq_len = len(seq)
        _, fasta_hash = _read_fasta_and_hash(fasta_path)

        if seq_len > LONG_SEQ_WARN_THRESHOLD and "attention" in trace_mode:
            log(
                f"Warning: sequence length {seq_len} > {LONG_SEQ_WARN_THRESHOLD}. "
                "Attention storage is N^2; consider --layers 0,1 or --trace_mode activations."
            )

        model = self.load_model()
        want_attn = "attention" in trace_mode
        want_act = "activations" in trace_mode
        layer_list = _parse_layers_arg(layers)
        head_list = _parse_heads_arg(heads)

        collector = ESMFoldTraceCollector(
            want_attention=want_attn,
            want_activations=want_act,
            layer_indices=layer_list,
            head_indices=head_list,
        )

        # Try model forward with optional outputs
        with torch.no_grad():
            # ESMFold in fair-esm: tokenize with model's alphabet
            alphabet = self._alphabet
            if alphabet is None:
                alphabet = esm.Alphabet.from_architecture("ESM-1b")
            batch_converter = alphabet.get_batch_converter()
            _, _, batch_tokens = batch_converter([(seq_id, seq)])
            batch_tokens = batch_tokens.to(device=self.device)

            kwargs = {}
            if hasattr(model, "forward"):
                sig = getattr(model.forward, "__wrapped__", model.forward)
                try:
                    import inspect
                    params = inspect.signature(model.forward).parameters
                    if "output_attentions" in params:
                        kwargs["output_attentions"] = want_attn
                    if "output_hidden_states" in params:
                        kwargs["output_hidden_states"] = want_act
                except Exception:
                    pass

            out = model(batch_tokens, **kwargs)

            got_attn, got_act = collector.try_use_outputs(out, self.model_name)
            if (want_attn and not got_attn) or (want_act and not got_act):
                collector.register_hooks(model)
                collector.clear()
                out = model(batch_tokens)
                collector.try_use_outputs(out, self.model_name)
                collector.remove_hooks()

        # Structure output
        pdb_str = None
        coords = None
        if hasattr(out, "positions"):
            pos = out.positions
            if pos is not None:
                coords = pos
        if hasattr(out, "to_pdb"):
            try:
                pdb_str = out.to_pdb(out)[0] if hasattr(out.to_pdb(out), "__getitem__") else out.to_pdb(out)
            except Exception:
                pass
        if pdb_str is None and hasattr(out, "pdb_string"):
            pdb_str = getattr(out, "pdb_string", None)
        if pdb_str is None and coords is not None:
            pdb_str = _coords_to_minimal_pdb(coords, seq)
        if pdb_str is None:
            log("Warning: no PDB output from model; structure/ may be incomplete.")

        struct_paths = write_structure(out_dir, pdb_str, coords)
        log(f"Structure written: {struct_paths}")

        # Traces
        shapes_recorded = {"attention": {}, "activations": {}}
        if trace_mode != "none" and (collector.attention or collector.activations):
            attn_idx, act_idx = write_traces(
                out_dir,
                collector,
                save_fp16=save_fp16,
                layer_indices=layer_list,
                head_indices=head_list,
            )
            for k, v in attn_idx.items():
                shapes_recorded["attention"][k] = v.get("shape", [])
            for k, v in act_idx.items():
                shapes_recorded["activations"][k] = v.get("shape", [])
            try:
                write_trace_summary(out_dir, collector)
            except Exception:
                pass

        layer_count = max(
            len(collector.attention),
            len(collector.activations),
            1,
        )
        head_count = 0
        if collector.attention:
            first_attn = next(iter(collector.attention.values()))
            if first_attn.dim() >= 3:
                head_count = first_attn.shape[-3] if first_attn.dim() == 3 else first_attn.shape[1]

        build_and_write_meta(
            out_dir=out_dir,
            model_name=self.model_name,
            fasta_path=os.path.abspath(fasta_path),
            device=self.device,
            dtype=self.dtype,
            seq_len=seq_len,
            fasta_hash=fasta_hash,
            layer_count=layer_count,
            head_count=head_count,
            trace_mode=trace_mode,
            shapes_recorded=shapes_recorded,
            seed=self.seed,
            deterministic=self.deterministic,
            save_fp16=save_fp16,
        )
        log("meta.json written.")

        return {
            "structure": struct_paths,
            "out_dir": out_dir,
            "trace_mode": trace_mode,
        }


def _coords_to_minimal_pdb(coords: torch.Tensor, seq: str) -> str:
    """Write minimal CA-only PDB from coords [N, 3] or [N, 37, 3]."""
    if coords.dim() == 3:
        ca = coords[:, 1, :]  # assume atom37 order: N, CA, C, ...
    else:
        ca = coords
    lines = []
    for i in range(ca.shape[0]):
        a = ca[i].cpu().numpy()
        res = seq[i] if i < len(seq) else "X"
        lines.append(
            f"ATOM  {i+1:5d}  CA  {res} A{i+1:4d}    {a[0]:8.3f}{a[1]:8.3f}{a[2]:8.3f}  1.00  0.00           C"
        )
    return "\n".join(lines) + "\n"
