"""
ESMFold inference via OpenFold model loading.

Strategy:
  1. Download the ESMFold checkpoint via torch.hub (if not already cached)
  2. Patch IPA keys on disk before loading — fair-esm's _load_model() validates
     key names at load time, so in-memory patching after the fact is too late.
  3. Call esm.pretrained.esmfold_v1() which now finds the patched checkpoint
  4. Enable attention capture via forward hooks (seq_attention) and native
     ATTENTION_METADATA injection (tri_att_start)
  5. Write structure/, meta.json, logs.txt + attention text files

Attention types captured:
  - msa_row_attn (sequence self-attention): 48 layers, 32 heads, [L x L]
  - triangle_start_attn (pair triangle attention): 48 layers, 4 heads, [L x L]
"""
import logging
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from vizfold.backends.esmfold.schema import _read_fasta_and_hash
from vizfold.backends.esmfold.trace_adapter import (
    build_and_write_meta,
    write_structure,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ESMFOLD_LAYER_COUNT = 48  # ESMFold v1 trunk layers

# IPA keys that need patching for VizFold compatibility.
# The checkpoint ships with `<key>.weight` but OpenFold v2 expects `<key>.linear.weight`.
# See: Meta ESM Issue #435
IPA_KEYS_TO_PATCH = [
    "trunk.structure_module.ipa.linear_q_points",
    "trunk.structure_module.ipa.linear_kv_points",
]

ESMFOLD_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt"
ESMFOLD_CHECKPOINT_NAME = "esmfold_3B_v1.pt"


def _get_checkpoint_path() -> str:
    """Return the path where fair-esm caches the ESMFold checkpoint."""
    hub_dir = torch.hub.get_dir()
    return os.path.join(hub_dir, "checkpoints", ESMFOLD_CHECKPOINT_NAME)


def _ensure_checkpoint_patched() -> None:
    """
    Download the ESMFold checkpoint if needed, then patch IPA keys on disk.

    Must be called before esm.pretrained.esmfold_v1() because fair-esm's
    _load_model() validates key names at load time. Idempotent — no-op if
    the checkpoint is already patched.
    """
    checkpoint_path = _get_checkpoint_path()

    if not os.path.exists(checkpoint_path):
        logger.info(f"Downloading ESMFold checkpoint to {checkpoint_path} ...")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.hub.download_url_to_file(
            ESMFOLD_CHECKPOINT_URL,
            checkpoint_path,
            progress=True,
        )

    logger.info("Checking ESMFold checkpoint IPA key format...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    needs_patch = any(
        f"{key}.weight" in state_dict for key in IPA_KEYS_TO_PATCH
    )
    if not needs_patch:
        logger.info("Checkpoint already uses VizFold-compatible key format — no patch needed.")
        return

    logger.info("Patching ESMFold checkpoint IPA keys on disk...")
    ckpt["model"] = _patch_esmfold_state_dict(state_dict)
    torch.save(ckpt, checkpoint_path)
    logger.info(f"Patched checkpoint saved to {checkpoint_path}")


def _patch_esmfold_state_dict(state_dict: dict) -> dict:
    """
    Rename IPA linear projection keys for VizFold/OpenFold compatibility.
    Returns the patched state dict.
    """
    patched_count = 0
    for key in IPA_KEYS_TO_PATCH:
        old_weight = f"{key}.weight"
        old_bias = f"{key}.bias"
        new_weight = f"{key}.linear.weight"
        new_bias = f"{key}.linear.bias"

        if old_weight in state_dict:
            state_dict[new_weight] = state_dict.pop(old_weight)
            state_dict[new_bias] = state_dict.pop(old_bias)
            patched_count += 1
            logger.info(f"  Patched IPA key: {key}")

    if patched_count > 0:
        logger.info(f"Patched {patched_count} IPA keys for VizFold compatibility")
    else:
        logger.info("No IPA keys needed patching (already compatible)")

    return state_dict


def _parse_layers_arg(layers_arg: Optional[str]) -> Optional[List[int]]:
    """Parse --layers argument: 'all' or '0,1,2' or '0:12'."""
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


def read_fasta(fasta_path: str) -> Tuple[str, str]:
    """Return (sequence, seq_id) from a FASTA file."""
    seq, _ = _read_fasta_and_hash(fasta_path)
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                return seq, line[1:].strip().split()[0]
    return seq, "seq"


class ESMFoldRunner:
    """
    Runs ESMFold inference through OpenFold's model architecture.

    Uses fair-esm to download the ESMFold checkpoint, patches IPA keys
    for VizFold compatibility, then runs inference. When trace_mode is
    'attention', captures attention maps from both sequence self-attention
    (via forward hook) and triangle attention (via ATTENTION_METADATA),
    saving them in the standard VizFold text-file format.

    Usage:
        runner = ESMFoldRunner(device="cuda")
        runner.run(
            fasta_path="examples/monomer/fasta_dir_6KWC/6KWC.fasta",
            out_dir="outputs/esmf_6KWC",
            trace_mode="attention",
        )
    """

    def __init__(
        self,
        device: str = "cpu",
        seed: Optional[int] = None,
        deterministic: bool = False,
    ):
        self.model_name = "esmfold_v1"
        self.device = device
        self.seed = seed
        self.deterministic = deterministic
        self._model = None
        self._hook_handles: List[Any] = []

    def load_model(self) -> Any:
        """Load ESMFold model via fair-esm, returning the model object."""
        if self._model is not None:
            return self._model

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        if self.deterministic:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
            warnings.warn("Deterministic mode may reduce speed.", UserWarning)

        try:
            import esm  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "ESMFold backend requires fair-esm. Install with:\n"
                "  pip install fair-esm\n"
                "See: https://github.com/facebookresearch/esm"
            )

        # Patch checkpoint on disk BEFORE calling esmfold_v1() — fair-esm's
        # _load_model() validates key names at load time with no interception point.
        _ensure_checkpoint_patched()

        logger.info("Loading ESMFold model via fair-esm...")
        self._model = esm.pretrained.esmfold_v1()
        self._model = self._model.eval().to(self.device)
        logger.info(f"ESMFold model loaded on {self.device}")
        return self._model

    def run(
        self,
        fasta_path: str,
        out_dir: str,
        trace_mode: str = "attention",
        layers: Optional[str] = None,
        top_k: int = 50,
        triangle_residue_idx: Optional[int] = None,
        attn_map_dir: Optional[str] = None,
        num_recycles_save: Optional[int] = None,
        log_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run ESMFold inference and write structure + optional attention traces.

        Args:
            fasta_path: Path to FASTA file (single sequence).
            out_dir: Output directory (will be created).
            trace_mode: 'attention' or 'none'.
            layers: Which layers to save: 'all' or '0,1,2' or '0:12'.
            top_k: Number of top attention values to save per head.
            triangle_residue_idx: Residue index for triangle attention slicing.
                If None, triangle attention is averaged across all starting residues.
            attn_map_dir: Directory for attention text files. Defaults to
                out_dir/attention_files.
            num_recycles_save: Reserved for future use.
            log_path: Path for log file. Defaults to out_dir/logs.txt.

        Returns:
            Dict with structure paths, out_dir, and trace_mode.
        """
        os.makedirs(out_dir, exist_ok=True)
        if log_path is None:
            log_path = os.path.join(out_dir, "logs.txt")
        if attn_map_dir is None:
            attn_map_dir = os.path.join(out_dir, "attention_files")

        def log(msg: str) -> None:
            with open(log_path, "a") as f:
                f.write(msg + "\n")
            logger.info(msg)

        seq, seq_id = read_fasta(fasta_path)
        seq_len = len(seq)
        _, fasta_hash = _read_fasta_and_hash(fasta_path)
        log(f"Sequence: {seq_id} ({seq_len} residues)")

        model = self.load_model()

        want_attn = trace_mode == "attention"
        layer_list = _parse_layers_arg(layers)

        if want_attn:
            os.makedirs(attn_map_dir, exist_ok=True)
            log(f"Attention capture enabled. Output dir: {attn_map_dir}")
            log(f"Top K: {top_k}")
            if layer_list:
                log(f"Saving layers: {layer_list}")
            else:
                log("Saving all layers")
            self._enable_attention_capture(model, attn_map_dir, top_k,
                                            triangle_residue_idx,
                                            num_recycles_save)

        log(f"Running ESMFold inference on {self.device}...")
        t_start = time.perf_counter()

        with torch.no_grad():
            pdb_output = model.infer_pdb(seq)

        inference_time = time.perf_counter() - t_start
        log(f"Inference time: {inference_time:.2f}s")

        if want_attn:
            self._save_captured_attention(attn_map_dir, top_k,
                                           triangle_residue_idx, layer_list, log)
            for h in self._hook_handles:
                h.remove()
            self._hook_handles.clear()

        struct_paths = write_structure(out_dir, pdb_output)
        log(f"Structure written: {struct_paths}")

        shapes_recorded = {}
        if want_attn and os.path.exists(attn_map_dir):
            attn_files = [f for f in os.listdir(attn_map_dir) if f.endswith(".txt")]
            shapes_recorded["attention_files"] = attn_files

        build_and_write_meta(
            out_dir=out_dir,
            model_name=self.model_name,
            fasta_path=os.path.abspath(fasta_path),
            device=self.device,
            dtype="float32",
            seq_len=seq_len,
            fasta_hash=fasta_hash,
            layer_count=ESMFOLD_LAYER_COUNT,
            head_count=None,
            trace_mode=trace_mode,
            shapes_recorded=shapes_recorded,
            seed=self.seed,
            deterministic=self.deterministic,
            top_k=top_k,
            triangle_residue_idx=triangle_residue_idx,
        )
        log("meta.json written.")

        return {
            "structure": struct_paths,
            "out_dir": out_dir,
            "trace_mode": trace_mode,
            "attn_map_dir": attn_map_dir if want_attn else None,
        }

    def _enable_attention_capture(
        self,
        model: Any,
        attn_map_dir: str,
        top_k: int,
        triangle_residue_idx: Optional[int],
        num_recycles_save: Optional[int],
    ) -> None:
        """
        Enable attention capture on the ESMFold model's folding trunk.

        ESMFold's FoldingTrunk has 48 TriangularSelfAttentionBlock layers.
        Each block has two captured attention types:

        1. seq_attention (esm.esmfold.v1.misc.Attention):
           fair-esm's own class. Returns (output, attn_weights) where
           attn_weights is [B, L, H, L] (einops rearrangement of [B, H, L, L]).
           Captured via forward hook; permuted to [B, H, L, L] for storage.
           Key format: "msa_attention_block_{i}_attn_0" -> msa_row_attn

        2. tri_att_start (openfold TriangleAttentionStartingNode):
           Uses VizFold's primitives.Attention internally. Captured by setting
           _attention_name and attention_config on the inner mha module,
           which _attention() reads to populate ATTENTION_METADATA.
           Key format: "tri_attention_block_{i}_attn_0" -> triangle_start_attn
        """
        from openfold.model.primitives import ATTENTION_METADATA

        if not hasattr(ATTENTION_METADATA, "recent_attention"):
            ATTENTION_METADATA.recent_attention = {}
        ATTENTION_METADATA.recent_attention.clear()

        attention_config = {
            "demo_attn": True,
            "triangle_residue_idx": triangle_residue_idx,
        }

        if not (hasattr(model, "trunk") and hasattr(model.trunk, "blocks")):
            logger.warning(
                "ESMFold model does not have expected trunk.blocks structure. "
                "Attention capture may not work."
            )
            return

        self._hook_handles = []

        for block_idx, block in enumerate(model.trunk.blocks):
            if hasattr(block, "seq_attention"):
                layer_name = f"msa_attention_block_{block_idx}_attn_0"

                def make_seq_hook(lname: str):
                    def hook(module, inp, output):
                        if not (isinstance(output, (tuple, list)) and len(output) >= 2):
                            return
                        weights = output[1]
                        if weights is None:
                            return
                        # fair-esm returns [B, L, H, L]; permute to [B, H, L, L]
                        if weights.dim() == 4:
                            weights = weights.permute(0, 2, 1, 3)
                        arr = weights.detach().cpu().numpy()
                        ATTENTION_METADATA.recent_attention.setdefault(lname, []).append(arr)
                    return hook

                handle = block.seq_attention.register_forward_hook(make_seq_hook(layer_name))
                self._hook_handles.append(handle)

            tri_mod = getattr(block, "tri_att_start", None)
            if tri_mod is not None:
                layer_name = f"tri_attention_block_{block_idx}_attn_0"
                mha = getattr(tri_mod, "mha", None)
                if mha is not None and hasattr(mha, "attention_config"):
                    mha.attention_config = attention_config
                    mha._attention_name = layer_name

        logger.info(
            f"Registered attention capture on {len(model.trunk.blocks)} trunk blocks "
            "(seq_attention hook + tri_att_start native)."
        )

    def _save_captured_attention(
        self,
        attn_map_dir: str,
        top_k: int,
        triangle_residue_idx: Optional[int],
        layer_list: Optional[List[int]],
        log,
    ) -> None:
        """
        Normalize captured attention shapes and save to text files.

        Both attention types accumulate one array per recycle. We take only the
        last recycle (most refined) and normalize shapes:

          msa_row_attn:       hook stores [B, H, L, L]  → (M, N, S, S) ✓
          triangle_start_attn: _attention() stores [B, L, H, L, L] (5D, pair dims);
                               squeeze B → [L, H, L, L] = (Q, R, S, S) ✓
        """
        from openfold.model.primitives import ATTENTION_METADATA
        from openfold.model.evoformer import (
            save_all_topk_from_recent_attention,
            parse_attention_metadata_key,
        )

        if not hasattr(ATTENTION_METADATA, "recent_attention"):
            log("WARNING: No attention data captured. ATTENTION_METADATA.recent_attention not found.")
            return

        if not ATTENTION_METADATA.recent_attention:
            log("WARNING: No attention data captured. recent_attention is empty.")
            return

        captured_keys = list(ATTENTION_METADATA.recent_attention.keys())
        log(f"Captured attention for {len(captured_keys)} layer(s): {captured_keys[:5]}...")

        normalized: dict = {}
        for k, v_list in ATTENTION_METADATA.recent_attention.items():
            if not v_list:
                continue

            attn_type, layer_idx = parse_attention_metadata_key(k)
            if attn_type is None or layer_idx < 0:
                continue

            if layer_list is not None and layer_idx not in layer_list:
                continue

            arr = v_list[-1]  # last recycle only

            if attn_type == "msa_row_attn":
                if arr.ndim == 4:
                    normalized[k] = arr
                else:
                    log(f"WARNING: unexpected msa_row_attn shape {arr.shape} for {k}, skipping")

            elif attn_type == "triangle_start_attn":
                if arr.ndim == 5:
                    normalized[k] = arr[0]  # squeeze batch dim
                elif arr.ndim == 4:
                    normalized[k] = arr
                else:
                    log(f"WARNING: unexpected triangle_start_attn shape {arr.shape} for {k}, skipping")

        if not normalized:
            log("WARNING: No valid attention data after normalization.")
            ATTENTION_METADATA.recent_attention.clear()
            return

        log(f"Normalized {len(normalized)} attention entries.")

        ATTENTION_METADATA.recent_attention = {k: [v] for k, v in normalized.items()}

        save_all_topk_from_recent_attention(
            save_dir=attn_map_dir,
            triangle_residue_idx=triangle_residue_idx,
            top_k=top_k,
        )
        log(f"Attention maps saved to {attn_map_dir}")

        ATTENTION_METADATA.recent_attention.clear()
