"""
ESMFold inference via OpenFold model loading.

Strategy:
  1. Download the ESMFold checkpoint via torch.hub (if not already cached)
  2. Patch IPA keys ON DISK before loading  ← must happen first
     fair-esm's _load_model() validates key names during load, so in-memory
     patching after the fact is too late.
  3. Call esm.pretrained.esmfold_v1() which now finds the patched checkpoint
  4. Run forward pass with demo_attn=True so ATTENTION_METADATA captures attention
  5. Write structure/, meta.json, logs.txt + attention text files

Note: ESMFold does NOT have triangle attention (no pair representation). Only
MSA row attention is available. This is an architectural difference from
OpenFold/AlphaFold2, not a code limitation.
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

# ESMFold model parameters
ESMFOLD_LAYER_COUNT = 48  # ESMFold v1 trunk layers
ESMFOLD_HEAD_COUNT = 8    # heads per attention layer

# IPA keys that need patching for VizFold compatibility.
# fair-esm downloads a checkpoint with these as `<key>.weight`,
# but OpenFold v2 (VizFold) renamed them to `<key>.linear.weight`.
# See: Meta ESM Issue #435
IPA_KEYS_TO_PATCH = [
    "trunk.structure_module.ipa.linear_q_points",
    "trunk.structure_module.ipa.linear_kv_points",
]

# Cache location where fair-esm stores the downloaded checkpoint
ESMFOLD_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt"
ESMFOLD_CHECKPOINT_NAME = "esmfold_3B_v1.pt"


def _get_checkpoint_path() -> str:
    """Return the path where fair-esm caches the ESMFold checkpoint."""
    hub_dir = torch.hub.get_dir()
    return os.path.join(hub_dir, "checkpoints", ESMFOLD_CHECKPOINT_NAME)


def _ensure_checkpoint_patched() -> None:
    """
    Download the ESMFold checkpoint if needed, then patch IPA keys on disk.

    This MUST be called before esm.pretrained.esmfold_v1() because fair-esm's
    _load_model() validates that expected key names are present and raises
    RuntimeError if they are missing — there is no way to intercept and fix
    keys after the fact.

    Idempotent: if the checkpoint is already patched, nothing is rewritten.
    """
    checkpoint_path = _get_checkpoint_path()

    # Download if not yet cached
    if not os.path.exists(checkpoint_path):
        logger.info(f"Downloading ESMFold checkpoint to {checkpoint_path} ...")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.hub.download_url_to_file(
            ESMFOLD_CHECKPOINT_URL,
            checkpoint_path,
            progress=True,
        )

    # Load and inspect — patch only if the OLD key format is present
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
    Patch ESMFold checkpoint keys for VizFold/OpenFold IPA compatibility.

    ESMFold uses `linear_q_points.weight` while OpenFold v2 (VizFold) expects
    `linear_q_points.linear.weight`. This patches the keys in-place.

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
    set to 'attention', leverages the existing ATTENTION_METADATA system
    in openfold.model.primitives to capture attention maps, which are
    saved in the standard VizFold text-file format.

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
            import esm  # noqa: F401 — checked before patching
        except ImportError:
            raise RuntimeError(
                "ESMFold backend requires fair-esm. Install with:\n"
                "  pip install fair-esm\n"
                "See: https://github.com/facebookresearch/esm"
            )

        # MUST patch the checkpoint on disk BEFORE calling esmfold_v1().
        # fair-esm's _load_model() raises RuntimeError if expected keys are
        # missing — there is no way to patch in memory after the fact.
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
            triangle_residue_idx: Residue index for triangle attention.
                NOTE: ESMFold has no triangle attention — this will be
                logged as a warning and ignored.
            attn_map_dir: Directory for attention text files. If None,
                defaults to out_dir/attention_files.
            num_recycles_save: Number of recycling iterations to save
                attention for. If None, saves last recycle only.
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

        # Handle triangle attention warning
        if triangle_residue_idx is not None:
            log(
                "WARNING: triangle_residue_idx was set but ESMFold does not "
                "have triangle attention (no pair representation). This "
                "parameter is ignored. Triangle attention is only available "
                "with OpenFold/AlphaFold2."
            )

        # Read FASTA
        seq, seq_id = read_fasta(fasta_path)
        seq_len = len(seq)
        _, fasta_hash = _read_fasta_and_hash(fasta_path)
        log(f"Sequence: {seq_id} ({seq_len} residues)")

        # Load model
        model = self.load_model()

        # Configure attention capture via ATTENTION_METADATA
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

            # Enable attention capture in the model
            # The ESMFold model uses OpenFold's EvoformerStack internally,
            # which reads attention_config from the model config
            self._enable_attention_capture(model, attn_map_dir, top_k,
                                            triangle_residue_idx,
                                            num_recycles_save)

        # Run inference
        log(f"Running ESMFold inference on {self.device}...")
        t_start = time.perf_counter()

        with torch.no_grad():
            pdb_output = model.infer_pdb(seq)

        inference_time = time.perf_counter() - t_start
        log(f"Inference time: {inference_time:.2f}s")

        # Save attention if captured
        if want_attn:
            self._save_captured_attention(attn_map_dir, top_k,
                                           triangle_residue_idx, layer_list, log)

        # Write structure
        struct_paths = write_structure(out_dir, pdb_output)
        log(f"Structure written: {struct_paths}")

        # Determine shapes for metadata
        shapes_recorded = {}
        if want_attn and os.path.exists(attn_map_dir):
            attn_files = [f for f in os.listdir(attn_map_dir) if f.endswith(".txt")]
            shapes_recorded["attention_files"] = attn_files

        # Write metadata
        build_and_write_meta(
            out_dir=out_dir,
            model_name=self.model_name,
            fasta_path=os.path.abspath(fasta_path),
            device=self.device,
            dtype="float32",
            seq_len=seq_len,
            fasta_hash=fasta_hash,
            layer_count=ESMFOLD_LAYER_COUNT,
            head_count=ESMFOLD_HEAD_COUNT,
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
        Enable the ATTENTION_METADATA capture system on the ESMFold model.

        This works by setting the attention_config on the model's internal
        EvoformerStack and its attention layers, which causes _attention()
        in primitives.py to save attention maps.
        """
        from openfold.model.primitives import ATTENTION_METADATA

        # Initialize ATTENTION_METADATA
        if not hasattr(ATTENTION_METADATA, "recent_attention"):
            ATTENTION_METADATA.recent_attention = {}
        ATTENTION_METADATA.recent_attention.clear()

        # Walk the model to find attention modules and set their config
        attention_config = {
            "demo_attn": True,
            "triangle_residue_idx": triangle_residue_idx,
        }

        # ESMFold's internal structure: model.trunk.evoformer (EvoformerStack)
        # We need to find and configure all attention layers
        for name, module in model.named_modules():
            # Set attention config on any module that accepts it
            if hasattr(module, "attention_config"):
                module.attention_config = attention_config
            # Also check for _attention_name (set by assign_layer_names)
            if hasattr(module, "mha") and hasattr(module.mha, "_attention_name"):
                pass  # Already configured by assign_layer_names

        # Try to call assign_layer_names if the model has EvoformerStack
        for name, module in model.named_modules():
            if hasattr(module, "assign_layer_names") and callable(module.assign_layer_names):
                module.assign_layer_names()
                logger.info(f"Assigned layer names on {name}")

        # Store config for later use
        if hasattr(model, "trunk") and hasattr(model.trunk, "evoformer"):
            model.trunk.evoformer.attn_map_dir = attn_map_dir

    def _save_captured_attention(
        self,
        attn_map_dir: str,
        top_k: int,
        triangle_residue_idx: Optional[int],
        layer_list: Optional[List[int]],
        log,
    ) -> None:
        """
        Save attention maps captured by ATTENTION_METADATA to text files.

        Uses the existing save_all_topk_from_recent_attention() from
        evoformer.py, which writes the standard VizFold text format.
        """
        from openfold.model.primitives import ATTENTION_METADATA
        from openfold.model.evoformer import save_all_topk_from_recent_attention

        if not hasattr(ATTENTION_METADATA, "recent_attention"):
            log("WARNING: No attention data captured. ATTENTION_METADATA.recent_attention not found.")
            return

        if not ATTENTION_METADATA.recent_attention:
            log("WARNING: No attention data captured. recent_attention is empty.")
            log("This may happen if the ESMFold model's internal architecture "
                "does not use the standard OpenFold attention path.")
            log("Consider using run_pretrained_openfold.py with an ESMFold "
                "checkpoint for full attention capture.")
            return

        captured_keys = list(ATTENTION_METADATA.recent_attention.keys())
        log(f"Captured attention for {len(captured_keys)} layer(s): {captured_keys[:5]}...")

        # Filter by layer list if specified
        if layer_list is not None:
            from openfold.model.evoformer import parse_attention_metadata_key
            filtered = {}
            for k, v in ATTENTION_METADATA.recent_attention.items():
                _, layer_idx = parse_attention_metadata_key(k)
                if layer_idx in layer_list:
                    filtered[k] = v
            ATTENTION_METADATA.recent_attention = filtered
            log(f"Filtered to {len(filtered)} layers: {layer_list}")

        # Save using existing VizFold pipeline
        save_all_topk_from_recent_attention(
            save_dir=attn_map_dir,
            triangle_residue_idx=triangle_residue_idx,
            top_k=top_k,
        )
        log(f"Attention maps saved to {attn_map_dir}")

        # Clean up
        ATTENTION_METADATA.recent_attention.clear()
