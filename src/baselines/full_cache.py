"""
Baseline 1: Full FP16 KV Cache (Uncompressed Ground Truth).
Wraps Hugging Face DynamicCache to preserve 100% of KV pairs.
"""

from typing import Optional, Tuple, Any, Dict, List
import torch

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    class DynamicCache:
        def __init__(self):
            self.key_cache_list = []
            self.value_cache_list = []

        def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if len(self.key_cache_list) <= layer_idx:
                self.key_cache_list.append(key_states)
                self.value_cache_list.append(value_states)
            else:
                self.key_cache_list[layer_idx] = torch.cat([self.key_cache_list[layer_idx], key_states], dim=-2)
                self.value_cache_list[layer_idx] = torch.cat([self.value_cache_list[layer_idx], value_states], dim=-2)
            return self.key_cache_list[layer_idx], self.value_cache_list[layer_idx]

        def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
            if not self.key_cache_list or layer_idx >= len(self.key_cache_list):
                return 0
            return self.key_cache_list[layer_idx].shape[-2]


class FullKVCacheWrapper(DynamicCache):
    """
    Standard Full KV Cache representing the upper-bound baseline (p_c = 0%).
    Keeps all key and value vectors in memory without any eviction or compression.
    """
    def __init__(self):
        super().__init__()
        self.compression_ratio: float = 0.0

    @property
    def name(self) -> str:
        return "Full-FP16-Cache"

    def reset_scores(self):
        """Reset cached states for a fresh sequence generation."""
        if hasattr(self, "layers"):
            self.layers = []
        if hasattr(self, "key_cache_list"):
            self.key_cache_list = []
        if hasattr(self, "value_cache_list"):
            self.value_cache_list = []

    @property
    def key_cache(self) -> List[torch.Tensor]:
        """Backwards compatibility accessor for key tensors."""
        if hasattr(self, "layers"):
            return [l.keys for l in self.layers if hasattr(l, "keys")]
        return getattr(self, "key_cache_list", [])

    @property
    def value_cache(self) -> List[torch.Tensor]:
        """Backwards compatibility accessor for value tensors."""
        if hasattr(self, "layers"):
            return [l.values for l in self.layers if hasattr(l, "values")]
        return getattr(self, "value_cache_list", [])

    def get_memory_bytes(self) -> int:
        """Calculate total bytes occupied by KV tensors in memory."""
        total_bytes = 0
        if hasattr(self, "layers"):
            for layer in self.layers:
                if hasattr(layer, "keys") and layer.keys is not None and layer.keys.numel() > 0:
                    total_bytes += layer.keys.numel() * layer.keys.element_size()
                if hasattr(layer, "values") and layer.values is not None and layer.values.numel() > 0:
                    total_bytes += layer.values.numel() * layer.values.element_size()
            return total_bytes

        for k in self.key_cache:
            if k is not None:
                total_bytes += k.numel() * k.element_size()
        for v in self.value_cache:
            if v is not None:
                total_bytes += v.numel() * v.element_size()
        return total_bytes
