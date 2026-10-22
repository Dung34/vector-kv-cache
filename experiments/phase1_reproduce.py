import os
import sys

# Đảm bảo in utf-8 trên Windows console
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Đảm bảo import được module từ src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import argparse
import datetime
import hashlib
import traceback
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from src.baselines.h2o_cache import H2OKVCache



def compute_file_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Reproduce Invariant Violations (I1 & I2)")
    parser.add_argument("--nonce", type=str, default="0", help="Nonce provided by user for verification")
    args = parser.parse_args()

    script_path = os.path.abspath(__file__)
    script_sha256 = compute_file_sha256(script_path)

    # Dòng đầu log luôn là header do script tự in theo Giao thức Bằng chứng
    print("=== HEADER ===")
    print(f"nonce: {args.nonce}")
    print(f"timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"torch_version: {torch.__version__}")
    try:
        import transformers
        tf_version = transformers.__version__
        print(f"transformers_version: {tf_version}")
    except Exception as e:
        print(f"transformers_version: unknown ({e})")
        tf_version = None

    # Version guard: dừng nếu không nằm trong {4.56.2, 5.20.1}
    SUPPORTED_VERSIONS = ("4.56.2", "5.20.1")
    if tf_version not in SUPPORTED_VERSIONS:
        raise RuntimeError(
            f"Unsupported transformers version: {tf_version}. Must be in {SUPPORTED_VERSIONS}"
        )

    patch_env = os.environ.get("H2O_DISABLE_MASK_PATCH", "0")
    patch_status = "DISABLED" if patch_env == "1" else "ENABLED"
    print(f"mask_patch_status: {patch_status} (H2O_DISABLE_MASK_PATCH={patch_env})")
    print(f"device: cpu")
    print(f"seed: 42")
    print(f"script_path: {script_path}")
    print(f"script_sha256: {script_sha256}")
    print("==============\n")

    torch.manual_seed(42)

    # 1. Khởi tạo mô hình Qwen2 config nhỏ trên CPU
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

    # 2. Khởi tạo H2OKVCache (context 64, budget 50% = 32 tokens)
    cache = H2OKVCache(compression_ratio=0.50, min_recent_tokens=8, num_sink_tokens=4)

    # 3. Prefill 64 tokens
    prefill_len = 64
    prompt = torch.randint(0, 1000, (1, prefill_len))
    cache.is_prefilling = True
    with torch.no_grad():
        _ = model(prompt, past_key_values=cache, use_cache=True, output_attentions=True)
    cache.is_prefilling = False

    history = []
    expected_seen_tokens = prefill_len

    # Thiết lập hook để ghi nhận chính xác dữ liệu đầu vào và đầu ra tại từng bước decode
    step_records = {}

    def make_pre_hook(layer_idx):
        def hook_fn(module, args, kwargs):
            step_records[f"layer_{layer_idx}_cpos"] = kwargs.get("cache_position")
            step_records[f"layer_{layer_idx}_posids"] = kwargs.get("position_ids")
            attn_mask = kwargs.get("attention_mask")
            step_records[f"layer_{layer_idx}_mask"] = attn_mask
            step_records[f"layer_{layer_idx}_mask_len"] = attn_mask.shape[-1] if attn_mask is not None else 0
        return hook_fn

    hooks = [
        model.model.layers[i].register_forward_pre_hook(make_pre_hook(i), with_kwargs=True)
        for i in range(2)
    ]

    # Chạy 2 bước decode
    for decode_step in [1, 2]:
        step_records.clear()
        past_seen_before_forward = cache.get_seq_length(0)
        token = torch.randint(0, 1000, (1, 1))

        with torch.no_grad():
            _ = model(token, past_key_values=cache, use_cache=True, output_attentions=True)

        physical_len_after = cache.key_cache[0].shape[-2]
        scores_len = cache.accumulated_scores[0].shape[-1] if len(cache.accumulated_scores) > 0 and cache.accumulated_scores[0] is not None else 0
        pos_id = step_records.get("layer_0_posids")
        pos_id_val = pos_id[0, -1].item() if pos_id is not None and pos_id.numel() > 0 else -1
        mask_len = step_records.get("layer_0_mask_len", 0)

        # Trích xuất shape/min/max thật của mask từ hook cho cả 2 layer
        m0 = step_records.get("layer_0_mask")
        m1 = step_records.get("layer_1_mask")
        m0_min = m0.min().item() if m0 is not None else float("nan")
        m0_max = m0.max().item() if m0 is not None else float("nan")
        m1_min = m1.min().item() if m1 is not None else float("nan")
        m1_max = m1.max().item() if m1 is not None else float("nan")

        history.append({
            "step": decode_step,
            "expected_seen": expected_seen_tokens,
            "past_seen": past_seen_before_forward,
            "physical_len": physical_len_after,
            "mask_kv_len": mask_len,
            "position_id": pos_id_val,
            "scores_len": scores_len,
            "m0_shape": tuple(m0.shape) if m0 is not None else None,
            "m0_min": m0_min,
            "m0_max": m0_max,
            "m1_shape": tuple(m1.shape) if m1 is not None else None,
            "m1_min": m1_min,
            "m1_max": m1_max,
        })

        expected_seen_tokens += 1

    for h in hooks:
        h.remove()

    # In giá trị mask thực tế từ hook ở từng layer từng bước
    print("--- THONG TIN MASK THUC TE (TU HOOK) ---")
    for row in history:
        s = row["step"]
        print(f"Buoc {s} Layer 0 mask: shape={row['m0_shape']} min={row['m0_min']} max={row['m0_max']}")
        print(f"Buoc {s} Layer 1 mask: shape={row['m1_shape']} min={row['m1_min']} max={row['m1_max']}")
    print("")

    # In bảng dữ liệu được TÍNH TOÁN thực tế từ tensor, không vẽ tay số liệu
    print("--- THUC DO BANG DU LIEU CAC BUOC DECODE ---")
    print(f"{'buoc':<6} | {'expected_seen':<14} | {'past_seen':<10} | {'physical_len':<13} | {'mask_kv_len':<12} | {'position_id':<12} | {'scores_len':<10}")
    print("-" * 88)
    for row in history:
        print(f"{row['step']:<6} | {row['expected_seen']:<14} | {row['past_seen']:<10} | {row['physical_len']:<13} | {row['mask_kv_len']:<12} | {row['position_id']:<12} | {row['scores_len']:<10}")
    print("-" * 88 + "\n")

    # Kiểm tra bất biến I1 và I2:
    # Ở bước 2:
    # expected position_id = 65 (seen_tokens trước bước đó: 64 + 1 = 65)
    # expected mask_kv_len = 32 (K sau update tỉa về active_budget = 32)
    step2 = history[1]
    
    # In phép so sánh cả hai vế theo quy tắc D: expected=<giá trị> actual=<giá trị>
    print(f"Kiem tra I1 (Step 2 position_id): expected={step2['expected_seen']} actual={step2['position_id']}")
    print(f"Kiem tra I2 (Step 2 mask_kv_len): expected={step2['physical_len']} actual={step2['mask_kv_len']}")

    try:
        assert step2["position_id"] == step2["expected_seen"], (
            f"BAT BIEN I1 BI VI PHAM: expected={step2['expected_seen']} actual={step2['position_id']}"
        )
        assert step2["mask_kv_len"] == step2["physical_len"], (
            f"BAT BIEN I2 BI VI PHAM: expected={step2['physical_len']} actual={step2['mask_kv_len']}"
        )
    except AssertionError:
        print("\n[TEST RESULT: FAILED THEO KY VONG]")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1)

    print("\n[TEST RESULT: PASSED]")
    sys.stdout.flush()
    sys.exit(0)





if __name__ == "__main__":
    main()
