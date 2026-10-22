"""
Patch utility to ensure Hugging Face Transformers attention forward functions
seamlessly support dynamic KV cache eviction and pruning (H2O, VectorKV, StreamingLLM).
"""

slice_counter = 0

def get_slice_count() -> int:
    global slice_counter
    return slice_counter

def reset_slice_count():
    global slice_counter
    slice_counter = 0

def patch_hf_eager_attention_for_pruned_cache():
    """
    Patches transformers eager attention forward functions (Qwen2, Llama, Mistral)
    so that when a KV cache dynamically prunes/evicts tokens, any causal attention_mask
    is dynamically sliced along the KV dimension (-1) to match key_states.shape[-2],
    preventing dimension mismatch RuntimeErrors.
    """
    import os
    if os.environ.get("H2O_DISABLE_MASK_PATCH") == "1":
        return
    targets = [
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
    ]
    import importlib
    for mod_name in targets:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "eager_attention_forward"):
                orig_fn = mod.eager_attention_forward
                if not getattr(orig_fn, "_is_pruned_cache_patched", False):
                    def make_patched(orig):
                        def patched_eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
                            if attention_mask is not None and key is not None:
                                kv_len = key.shape[-2]
                                assert attention_mask.shape[-1] == kv_len, (
                                    f"Attention mask KV length ({attention_mask.shape[-1]}) does not match key length ({kv_len})"
                                )
                            return orig(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
                        patched_eager_attention_forward._is_pruned_cache_patched = True
                        return patched_eager_attention_forward

                    mod.eager_attention_forward = make_patched(orig_fn)
        except Exception as e:
            print(f"[ERR] type={type(e).__name__} msg={e}")
