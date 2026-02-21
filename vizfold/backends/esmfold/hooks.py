"""
Hook-based extraction of attention and hidden states from ESMFold.

Strategy:
  1. Prefer model outputs (output_attentions=True, output_hidden_states=True) if available.
  2. Fall back to forward hooks on known modules with a clear warning.
"""
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class ESMFoldTraceCollector:
    """
    Collects attention weights and/or hidden states during ESMFold forward.

    Can use either returned outputs (preferred) or registered hooks.
    """

    def __init__(
        self,
        want_attention: bool = True,
        want_activations: bool = True,
        layer_indices: Optional[List[int]] = None,
        head_indices: Optional[List[int]] = None,
    ):
        self.want_attention = want_attention
        self.want_activations = want_activations
        self.layer_indices = layer_indices  # None => all
        self.head_indices = head_indices
        self.attention: Dict[str, torch.Tensor] = {}
        self.activations: Dict[str, torch.Tensor] = {}
        self._handles: List[Any] = []

    def clear(self) -> None:
        self.attention.clear()
        self.activations.clear()

    def _store_attention(self, name: str, attn: torch.Tensor, layer_idx: int) -> None:
        if self.layer_indices is not None and layer_idx not in self.layer_indices:
            return
        key = f"layer_{layer_idx:03d}"
        if key not in self.attention:
            self.attention[key] = attn.detach()
        else:
            self.attention[key] = torch.stack([self.attention[key], attn.detach()], dim=0)

    def _store_activation(self, name: str, h: torch.Tensor, layer_idx: int) -> None:
        if self.layer_indices is not None and layer_idx not in self.layer_indices:
            return
        key = f"layer_{layer_idx:03d}"
        self.activations[key] = h.detach()

    def try_use_outputs(
        self,
        outputs: Any,
        model_name: str = "esmfold",
    ) -> Tuple[bool, bool]:
        """
        Try to populate from model forward return value.
        Returns (got_attention, got_activations).
        """
        got_attn, got_act = False, False
        if hasattr(outputs, "attentions") and outputs.attentions is not None and self.want_attention:
            for i, attn in enumerate(outputs.attentions):
                if attn is not None:
                    self._store_attention("output", attn, i)
                    got_attn = True
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None and self.want_activations:
            for i, h in enumerate(outputs.hidden_states):
                if h is not None:
                    self._store_activation("output", h, i)
                    got_act = True
        if self.want_attention and not got_attn and hasattr(outputs, "states"):
            # ESMFold may return trunk states; no standard attentions
            pass
        return got_attn, got_act

    def register_hooks(self, model: nn.Module) -> None:
        """
        Register forward hooks to capture attention/activations if not from outputs.
        ESMFold structure: esm trunk -> folding trunk. We hook transformer layers.
        """
        layer_idx = [0]

        def _make_attn_hook(idx: int) -> Callable:
            def hook(module: nn.Module, inp: Any, out: Any) -> None:
                if isinstance(out, tuple):
                    # Many attention modules return (output, attn_weights)
                    for o in out:
                        if o is not None and o.dim() >= 3 and o.shape[0] == out[0].shape[0]:
                            self._store_attention("hook", o, idx)
                            break
                elif out is not None and out.dim() >= 3:
                    self._store_attention("hook", out, idx)
            return hook

        def _make_act_hook(idx: int) -> Callable:
            def hook(module: nn.Module, inp: Any, out: Any) -> None:
                if isinstance(out, tuple):
                    out = out[0]
                if out is not None and out.dim() >= 2:
                    self._store_activation("hook", out, idx)
            return hook

        for name, module in model.named_modules():
            if "attention" in name.lower() and hasattr(module, "forward"):
                try:
                    h = module.register_forward_hook(_make_attn_hook(layer_idx[0]))
                    self._handles.append(h)
                    layer_idx[0] += 1
                except Exception:
                    pass
            if "layer" in name.lower() and "encoder" in name.lower() and hasattr(module, "forward"):
                try:
                    h = module.register_forward_hook(_make_act_hook(layer_idx[0]))
                    self._handles.append(h)
                    layer_idx[0] += 1
                except Exception:
                    pass

        if self._handles:
            warnings.warn(
                "ESMFold: using forward hooks for trace extraction; "
                "output_attentions/output_hidden_states were not available.",
                UserWarning,
            )

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
