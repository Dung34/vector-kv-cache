import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import inspect
import datetime
import hashlib
import argparse
import torch
import transformers

def compute_file_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", type=str, default="1001")
    args = parser.parse_args()

    script_path = os.path.abspath(__file__)
    script_sha256 = compute_file_sha256(script_path)

    print("=== HEADER ===")
    print(f"nonce: {args.nonce}")
    print(f"timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"torch_version: {torch.__version__}")
    print(f"transformers_version: {transformers.__version__}")
    print(f"device: cpu")
    print(f"seed: 42")
    print(f"script_path: {script_path}")
    print(f"script_sha256: {script_sha256}")
    print("==============\n")

    # Mục a: eager_attention_forward có cắt lát mask không
    print("=== MỤC A: EAGER_ATTENTION_FORWARD SOURCE ===")
    from transformers.models.qwen2 import modeling_qwen2
    fn = modeling_qwen2.eager_attention_forward
    fn_file = inspect.getfile(fn)
    lines, start = inspect.getsourcelines(fn)
    print(f"File: {fn_file} (start line {start})")
    print(inspect.getsource(fn))
    print("============================================\n")

    # Mục b: get_seq_length / get_mask_sizes: chữ ký và nguồn
    print("=== MỤC B: CACHE GET_SEQ_LENGTH & GET_MASK_SIZES SOURCE ===")
    from transformers.cache_utils import Cache, DynamicCache
    print(f"Cache file: {inspect.getfile(Cache)}")
    print("--- DynamicCache.get_seq_length ---")
    print(inspect.getsource(DynamicCache.get_seq_length))
    print("--- DynamicCache.get_mask_sizes ---")
    if hasattr(DynamicCache, "get_mask_sizes"):
        print(inspect.getsource(DynamicCache.get_mask_sizes))
    else:
        print("DynamicCache does NOT have get_mask_sizes attribute")
    
    # Kiểm tra thuộc tính layers của Cache và DynamicCache
    dummy_cache = DynamicCache()
    print(f"DynamicCache has 'layers' attribute: {hasattr(dummy_cache, 'layers')}")
    if hasattr(dummy_cache, "layers"):
        print(f"Type of layers: {type(dummy_cache.layers)}")
    print(f"DynamicCache has 'key_cache' attribute: {hasattr(dummy_cache, 'key_cache')}")
    print(f"DynamicCache has '_key_cache' attribute: {hasattr(dummy_cache, '_key_cache')}")
    print("===========================================================\n")

    # Mục c: output[1] của self_attn ở prefill (type, shape)
    print("=== MỤC C: OUTPUT[1] CỦA SELF_ATTN Ở PREFILL ===")
    from transformers import Qwen2Config, Qwen2ForCausalLM
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

    intercepted = []
    def hook_fn(module, args, output):
        intercepted.append(output)

    h = model.model.layers[0].self_attn.register_forward_hook(hook_fn)
    prompt = torch.randint(0, 1000, (1, 64))
    with torch.no_grad():
        _ = model(prompt, output_attentions=False)
    h.remove()

    out_tuple = intercepted[0]
    print(f"self_attn output type: {type(out_tuple)}")
    print(f"self_attn output len: {len(out_tuple)}")
    print(f"output[0] type and shape: {type(out_tuple[0])} {out_tuple[0].shape}")
    print(f"output[1] type: {type(out_tuple[1])}")
    if isinstance(out_tuple[1], torch.Tensor):
        print(f"output[1] shape: {out_tuple[1].shape}")
    else:
        print(f"output[1] value: {out_tuple[1]}")
    print("===============================================\n")

if __name__ == "__main__":
    main()
