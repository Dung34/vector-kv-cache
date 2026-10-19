"""
Attention Hook Manager for VECTOR KV Cache Compression.
Intercepts attention weights during the forward pass and accumulates scores into VectorKVCache,
properly handling Grouped-Query Attention (GQA) architectures (e.g., Qwen2.5, Llama-3).
"""

from typing import Optional, List
import torch
import torch.nn as nn
from .vector_cache import VectorKVCache


class VectorAttentionHookManager:
    """
    Manages forward hooks on Transformer attention layers to capture attention weights
    and update accumulated scores in VectorKVCache in real-time.
    """

    def __init__(
        self,
        model: nn.Module,
        kv_cache: VectorKVCache,
        num_query_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
    ):
        self.model = model
        self.kv_cache = kv_cache
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Auto-detect head counts from model config if available
        config = getattr(model, "config", None)
        if config is not None:
            self.num_query_heads = num_query_heads or getattr(config, "num_attention_heads", 12)
            self.num_kv_heads = num_kv_heads or getattr(config, "num_key_value_heads", 2)
        else:
            self.num_query_heads = num_query_heads or 12
            self.num_kv_heads = num_kv_heads or 2

        self.group_size = max(1, self.num_query_heads // self.num_kv_heads)

    def _process_gqa_attention_scores(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Maps attention weights from Query heads to KV heads in GQA.
        attn_weights shape: [batch_size, num_query_heads, query_len, seq_len]
        Returns: [batch_size, num_kv_heads, query_len, seq_len]
        """
        if attn_weights.dim() == 4:
            batch_size, num_heads, q_len, kv_len = attn_weights.shape
            if num_heads == self.num_kv_heads:
                return attn_weights
            if num_heads == self.num_query_heads:
                # Group Query heads corresponding to the same KV head
                grouped = attn_weights.view(batch_size, self.num_kv_heads, self.group_size, q_len, kv_len)
                # Average attention across the query heads within the group
                return grouped.mean(dim=2)
        return attn_weights

    def register_hooks(self):
        """Register forward hooks onto all self_attn modules of the model."""
        self.remove_hooks()
        try:
            from src.utils.mask_patch import patch_hf_eager_attention_for_pruned_cache
            patch_hf_eager_attention_for_pruned_cache()
        except Exception:
            pass

        # Pre-hook on model to bypass causal mask creation during single-token decode
        def _model_decode_mask_pre_hook(module, args, kwargs):
            input_ids = kwargs.get("input_ids", None)
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
            if input_ids is not None and hasattr(input_ids, "shape") and input_ids.shape[-1] == 1:
                attn_mask = kwargs.get("attention_mask", None)
                if attn_mask is None or not isinstance(attn_mask, dict):
                    kwargs["attention_mask"] = {"full_attention": None}
            return args, kwargs

        try:
            handle = self.model.register_forward_pre_hook(_model_decode_mask_pre_hook, with_kwargs=True)
            self.hooks.append(handle)
        except Exception:
            pass

        # Find layers container (supports model.layers, model.model.layers, etc.)
        layers = None
        for path in ["model.layers", "layers", "transformer.h", "decoder.layers"]:
            curr = self.model
            for part in path.split("."):
                curr = getattr(curr, part, None)
                if curr is None:
                    break
            if curr is not None and isinstance(curr, (list, nn.ModuleList)):
                layers = curr
                break

        if layers is None:
            layers = [m for m in self.model.modules() if hasattr(m, "self_attn")]

        for layer_idx, layer_module in enumerate(layers):
            attn_module = getattr(layer_module, "self_attn", layer_module)

            def make_hook(idx: int):
                def hook_fn(module, args, output):
                    attn_weights = None
                    if isinstance(output, tuple) and len(output) > 1:
                        attn_weights = output[1]
                    elif hasattr(module, "last_attn_weights"):
                        attn_weights = module.last_attn_weights

                    if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                        processed_scores = self._process_gqa_attention_scores(attn_weights.detach())
                        self.kv_cache.register_attention_scores(idx, processed_scores)

                return hook_fn

            handle = attn_module.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks to avoid memory leaks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
