"""
Project III: Asymmetric KV Cache Compression for Small Language Models (SLMs)
Baseline Implementations: Full FP16 and H2O Binary Eviction
"""

__version__ = "0.1.0"

import os

if os.environ.get("H2O_DISABLE_MASK_PATCH") != "1":
    try:
        from .utils.mask_patch import patch_hf_eager_attention_for_pruned_cache
        patch_hf_eager_attention_for_pruned_cache()
    except Exception as e:
        print(f"[ERR] type={type(e).__name__} msg={e}")
