"""
Activation Collector Module.
Registers forward hooks on Transformer layers to stream (K, V) activations
during prefill forward passes and feed them directly into PerHeadOLSSolver.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from .rope_decoupler import decouple_rope


class ActivationCollector:
    """
    Hooks Qwen2 / Llama attention layers to extract Key and Value states.
    Can operate in two modes:
      1. Pre-RoPE extraction via k_proj & v_proj hooks (pure hidden state projections).
      2. Post-RoPE extraction + explicit decouple_rope transformation.
    """

    def __init__(
        self,
        model: nn.Module,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.hooks: List[Any] = []

        # Temporary storage per layer during forward pass
        self.captured_k: Dict[int, torch.Tensor] = {}
        self.captured_v: Dict[int, torch.Tensor] = {}

    def _find_attention_modules(self) -> List[Tuple[int, nn.Module]]:
        """Finds all attention blocks in the model."""
        modules = []
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for idx, layer in enumerate(self.model.model.layers):
                if hasattr(layer, "self_attn"):
                    modules.append((idx, layer.self_attn))
        return modules

    def register_hooks(self):
        """Attaches forward hooks to each layer's self_attn."""
        self.remove_hooks()
        attn_modules = self._find_attention_modules()

        for layer_idx, attn_mod in attn_modules:
            # We hook k_proj and v_proj directly for precise K_pre-RoPE and V extraction
            def make_k_hook(idx):
                def hook(module, inputs, output):
                    # output: [batch_size, seq_len, num_kv_heads * head_dim]
                    batch_size, seq_len, _ = output.shape
                    k = output.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    self.captured_k[idx] = k.detach()
                return hook

            def make_v_hook(idx):
                def hook(module, inputs, output):
                    # output: [batch_size, seq_len, num_kv_heads * head_dim]
                    batch_size, seq_len, _ = output.shape
                    v = output.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    self.captured_v[idx] = v.detach()
                return hook

            if hasattr(attn_mod, "k_proj") and hasattr(attn_mod, "v_proj"):
                h_k = attn_mod.k_proj.register_forward_hook(make_k_hook(layer_idx))
                h_v = attn_mod.v_proj.register_forward_hook(make_v_hook(layer_idx))
                self.hooks.extend([h_k, h_v])

    def remove_hooks(self):
        """Removes all active hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.captured_k.clear()
        self.captured_v.clear()

    def get_layer_kv(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns the captured (K, V) pair for a layer and removes it from buffer."""
        k = self.captured_k.pop(layer_idx, None)
        v = self.captured_v.pop(layer_idx, None)
        return k, v
