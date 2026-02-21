"""
ESMFold inference: load model, run forward, optionally extract traces.

Uses HuggingFace Transformers (EsmForProteinFolding) to avoid OpenFold build dependency.
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

# HuggingFace ESMFold
try:
    from transformers import AutoTokenizer, EsmForProteinFolding
    HAS_ESM = True
except ImportError:
    HAS_ESM = False
    AutoTokenizer = None  # type: ignore
    EsmForProteinFolding = None  # type: ignore

# PDB conversion (bundled in transformers)
def _get_pdb_utils():
    try:
        from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
        from transformers.models.esm.openfold_utils.protein import Protein as OFProtein, to_pdb
        return atom14_to_atom37, OFProtein, to_pdb
    except ImportError:
        return None, None, None


LONG_SEQ_WARN_THRESHOLD = 400  # N^2 attention warning
HF_MODEL_DEFAULT = "facebook/esmfold_v1"


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

    Uses HuggingFace EsmForProteinFolding (no fair-esm / OpenFold build).
    """

    def __init__(
        self,
        model_name: str = HF_MODEL_DEFAULT,
        device: str = "cpu",
        dtype: Optional[str] = None,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ):
        if not HAS_ESM:
            raise RuntimeError(
                "ESMFold backend requires transformers. Install with: pip install transformers torch"
            )
        self.model_name = model_name
        self.device = device
        self.dtype = dtype or "float32"
        self.seed = seed
        self.deterministic = deterministic
        self._model = None
        self._tokenizer = None

    def load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self.seed is not None:
            torch.manual_seed(self.seed)
        if self.deterministic:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
            warnings.warn("Deterministic mode may reduce speed.", UserWarning)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = EsmForProteinFolding.from_pretrained(self.model_name)
        self._model = self._model.eval()
        dtype_t = torch.float16 if self.dtype == "float16" else torch.float32
        self._model = self._model.to(device=self.device, dtype=dtype_t)
        return self._model

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
        tokenizer = self._tokenizer
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

        # Tokenize: single sequence, no special tokens (per HF ESMFold usage)
        inputs = tokenizer(
            [seq],
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
        )
        input_ids = inputs["input_ids"].to(device=self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.device)

        with torch.no_grad():
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "output_attentions": want_attn,
                "output_hidden_states": want_act,
            }
            out = model(**kwargs)

        got_attn, got_act = collector.try_use_outputs(out, "esmfold")
        if (want_attn and not got_attn) or (want_act and not got_act):
            # Capture from ESM trunk (model.esm) via hooks
            esm_trunk = getattr(model, "esm", model)
            collector.register_hooks(esm_trunk)
            collector.clear()
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            collector.try_use_outputs(out, "esmfold")
            collector.remove_hooks()

        # Structure: PDB + optional coords tensor (HF returns dict-like or object)
        pdb_str = None
        coords = None
        atom14_to_atom37, OFProtein, to_pdb = _get_pdb_utils()

        def _get_out(key: str):
            if hasattr(out, key):
                return getattr(out, key)
            if isinstance(out, dict) and key in out:
                return out[key]
            return None

        positions = _get_out("positions")
        if positions is not None:
            try:
                if atom14_to_atom37 and OFProtein is not None and to_pdb is not None:
                    # positions: often [num_recycling, batch, N, 14, 3]; use last recycling
                    pos = positions
                    if pos.dim() == 5:
                        pos = pos[-1]
                    final_atom37 = atom14_to_atom37(pos, out)
                    coords = final_atom37
                    out_cpu = {}
                    for k in ("aatype", "atom37_atom_exists", "residue_index", "plddt", "chain_index"):
                        v = _get_out(k)
                        if v is not None:
                            out_cpu[k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                    pos_np = final_atom37.cpu().numpy()
                    for i in range(pos_np.shape[0]):
                        pred = OFProtein(
                            aatype=out_cpu["aatype"][i],
                            atom_positions=pos_np[i],
                            atom_mask=out_cpu["atom37_atom_exists"][i],
                            residue_index=out_cpu["residue_index"][i] + 1,
                            b_factors=out_cpu["plddt"][i],
                            chain_index=out_cpu.get("chain_index", [None] * pos_np.shape[0])[i] if "chain_index" in out_cpu else None,
                        )
                        pdb_str = to_pdb(pred)
                        break
                else:
                    pos = positions
                    if pos.dim() == 5:
                        pos = pos[-1, 0]
                    elif pos.dim() == 4:
                        pos = pos[0]
                    coords = pos
                    if coords.dim() == 4:
                        ca_idx = 1
                        coords = coords[:, :, ca_idx, :]
                    pdb_str = _coords_to_minimal_pdb(coords[0], seq)
            except Exception as e:
                log(f"Warning: PDB conversion failed ({e}); writing minimal PDB if possible.")
                if coords is not None:
                    try:
                        c = coords[0] if coords.dim() > 2 else coords
                        if c.dim() == 3:
                            c = c[:, 1, :]
                        pdb_str = _coords_to_minimal_pdb(c, seq)
                    except Exception:
                        pass

        if pdb_str is None and coords is not None:
            c = coords[0] if coords.dim() > 2 else coords
            if c.dim() == 3:
                c = c[:, 1, :]
            pdb_str = _coords_to_minimal_pdb(c, seq)
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
