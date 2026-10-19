"""
Project III: Asymmetric KV Cache Compression for Small Language Models (SLMs)
Baseline Implementations: Full FP16 and H2O Binary Eviction
"""

__version__ = "0.1.0"

try:
    from .utils.mask_patch import patch_hf_eager_attention_for_pruned_cache
    patch_hf_eager_attention_for_pruned_cache()
except Exception:
    pass
