import os
import sys
import argparse
import datetime
import hashlib
import inspect
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def compute_file_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", type=str, default="0")
    args = parser.parse_args()

    script_path = os.path.abspath(__file__)
    script_sha256 = compute_file_sha256(script_path)

    print("=== HEADER ===")
    print(f"nonce: {args.nonce}")
    print(f"timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"torch_version: {torch.__version__}")
    import transformers
    print(f"transformers_version: {transformers.__version__}")
    print(f"device: cpu")
    print(f"seed: 42")
    print(f"script_path: {script_path}")
    print(f"script_sha256: {script_sha256}")
    print("==============\n")

    # =========================================================================
    # MỤC 2: CƠ CHẾ BÁM CỦA PATCH TRÊN TRANSFORMERS 5.20.1
    # =========================================================================
    print("=== MỤC 2: CƠ CHẾ BÁM CỦA PATCH TRÊN TRANSFORMERS ===")
    import transformers.models.qwen2.modeling_qwen2 as qm
    fn_before = qm.eager_attention_forward
    id_before = id(fn_before)
    name_before = fn_before.__qualname__ if hasattr(fn_before, "__qualname__") else str(fn_before)
    print(f"Trước patch: qm.eager_attention_forward = {name_before}, id = {id_before}")

    # Áp dụng patch trực tiếp
    os.environ["H2O_DISABLE_MASK_PATCH"] = "0"
    from src.utils.mask_patch import patch_hf_eager_attention_for_pruned_cache
    patch_hf_eager_attention_for_pruned_cache()

    fn_after = qm.eager_attention_forward
    id_after = id(fn_after)
    name_after = fn_after.__qualname__ if hasattr(fn_after, "__qualname__") else str(fn_after)
    print(f"Sau patch:   qm.eager_attention_forward = {name_after}, id = {id_after}")
    print(f"Hàm bị thay thế: {name_before} -> {name_after}")
    print(f"id() thay đổi: {id_before != id_after} (id_before={id_before}, id_after={id_after})")
    print(f"Thuộc tính cờ _is_pruned_cache_patched: {getattr(fn_after, '_is_pruned_cache_patched', False)}")

    # Kiểm tra ALL_ATTENTION_FUNCTIONS get_interface
    interface_fn = qm.ALL_ATTENTION_FUNCTIONS.get_interface("eager", qm.eager_attention_forward)
    print(f"ALL_ATTENTION_FUNCTIONS.get_interface('eager', ...) trả về: {interface_fn.__qualname__}, id = {id(interface_fn)}")
    print("====================================================\n")

    # =========================================================================
    # MỤC 3: MRO VÀ THỰC TẾ GỌI GET_SEQ_LENGTH / GET_MASK_SIZES TRÊN CACHE
    # =========================================================================
    print("=== MỤC 3: MRO VÀ NGUỒN PHƯƠNG THỨC THỰC TẾ TRÊN H2OKVCache ===")
    from src.baselines.h2o_cache import H2OKVCache
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM

    print("H2OKVCache.__mro__:")
    for idx, cls in enumerate(H2OKVCache.__mro__):
        print(f"  [{idx}] {cls.__module__}.{cls.__qualname__}")

    cache = H2OKVCache(compression_ratio=0.5, min_recent_tokens=8, num_sink_tokens=4)
    print(f"\nKhởi tạo cache:")
    print(f"hasattr(cache, 'layers'): {hasattr(cache, 'layers')}")
    print(f"hasattr(cache, '_key_cache'): {hasattr(cache, '_key_cache')}")
    print(f"hasattr(cache, 'key_cache'): {hasattr(cache, 'key_cache')}")

    # Chạy prefill 64 tokens với mô hình nhỏ
    config = Qwen2Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        _attn_implementation="eager"
    )
    model = Qwen2ForCausalLM(config).eval()
    prompt = torch.randint(0, 1000, (1, 64))
    cache.is_prefilling = True
    with torch.no_grad():
        model(prompt, past_key_values=cache, use_cache=True, output_attentions=True)
    cache.is_prefilling = False

    print(f"\nSau prefill 64 tokens:")
    print(f"hasattr(cache, 'layers'): {hasattr(cache, 'layers')}")
    if hasattr(cache, "layers"):
        print(f"len(cache.layers): {len(cache.layers)}")
        print(f"type(cache.layers): {type(cache.layers)}")
    print(f"len(cache._key_cache): {len(cache._key_cache)}")
    print(f"cache._key_cache[0].shape: {cache._key_cache[0].shape}")
    print(f"cache.get_seq_length(0): {cache.get_seq_length(0)}")

    # Chạy 1 bước decode
    token = torch.randint(0, 1000, (1, 1))
    with torch.no_grad():
        model(token, past_key_values=cache, use_cache=True, output_attentions=True)

    print(f"\nSau bước decode 1 (đã prune về 32):")
    print(f"cache._key_cache[0].shape: {cache._key_cache[0].shape}")
    print(f"cache.get_seq_length(0): {cache.get_seq_length(0)}")
    print(f"cache.get_mask_sizes(1, 0): {cache.get_mask_sizes(1, 0)}")

    print("\n--- NGUỒN CỦA cache.get_seq_length (inspect.getsource trên instance) ---")
    print(inspect.getsource(cache.get_seq_length))

    print("--- NGUỒN CỦA cache.get_mask_sizes (inspect.getsource trên instance) ---")
    print(inspect.getsource(cache.get_mask_sizes))
    print("=================================================================\n")

if __name__ == "__main__":
    main()
