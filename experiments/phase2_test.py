import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import hashlib
from datetime import datetime, timezone

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

from src.baselines.h2o_cache import H2OKVCache


def get_file_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Phase 2 KV Cache 20-step verification test")
    parser.add_argument("--nonce", type=str, required=True, help="Nonce identifier")
    parser.add_argument("--steps", type=int, default=20, help="Number of decode steps (>=20)")
    args = parser.parse_args()

    script_sha = get_file_sha256(__file__)
    print(f"[HDR] nonce={args.nonce}")
    print(f"[HDR] time={datetime.now(timezone.utc).isoformat()}")
    print(f"[HDR] python={sys.version.split()[0]}")
    print(f"[HDR] torch={torch.__version__}")
    print(f"[HDR] transformers={transformers.__version__}")
    print(f"[HDR] device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"[HDR] seed=42")
    print(f"[HDR] script_sha256={script_sha}")

    torch.manual_seed(42)

    # Khởi tạo mô hình Qwen2 2-layer nhỏ
    config = AutoConfig.for_model(
        "qwen2",
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(config)
    model.eval()

    prompt_len = 64
    compression_ratio = 0.5
    cache = H2OKVCache(compression_ratio=compression_ratio)

    step_records = {}

    def make_pre_hook(layer_idx):
        def hook_fn(module, m_args, kwargs):
            if layer_idx == 0:
                step_records["posids"] = kwargs.get("position_ids")
                attn_mask = kwargs.get("attention_mask")
                step_records["mask_len"] = attn_mask.shape[-1] if attn_mask is not None else 0
        return hook_fn

    for i in range(2):
        model.model.layers[i].register_forward_pre_hook(make_pre_hook(i), with_kwargs=True)

    # Prefill
    prompt_ids = torch.randint(0, 1000, (1, prompt_len))
    with torch.no_grad():
        _ = model(prompt_ids, past_key_values=cache, use_cache=True, output_attentions=True)

    expected_budget = int(prompt_len * (1.0 - compression_ratio))
    print(f"[CFG] prompt_len={prompt_len} active_budget={cache.active_budget} expected_budget={expected_budget}")

    all_i1_passed = True
    all_i2_passed = True
    pos_id_sequence = []

    for step in range(1, args.steps + 1):
        step_records.clear()
        past_seen = cache.get_seq_length(0)
        token = torch.randint(0, 1000, (1, 1))

        try:
            with torch.no_grad():
                _ = model(token, past_key_values=cache, use_cache=True, output_attentions=True)
        except Exception as e:
            print(f"[ERR] step={step} type={type(e).__name__} msg={e}")
            print(f"[RESULT] FAIL reason=exception_at_step_{step}")
            sys.exit(1)

        phys_len = cache.key_cache[0].shape[-2]
        scores_len = cache.accumulated_scores[0].shape[-1] if len(cache.accumulated_scores) > 0 and cache.accumulated_scores[0] is not None else 0
        pos_id_tensor = step_records.get("posids")
        pos_id_val = pos_id_tensor[0, -1].item() if pos_id_tensor is not None and pos_id_tensor.numel() > 0 else -1
        mask_kv_len = step_records.get("mask_len", 0)
        expected_pos_id = prompt_len + step - 1  # 64, 65, 66, ...

        pos_id_sequence.append(pos_id_val)
        seen = cache.get_seq_length(0)

        print(f"[STEP] n={step} seen={seen} past_seen={past_seen} phys_len={phys_len} mask_kv={mask_kv_len} pos_id={pos_id_val} scores_len={scores_len}")

        # Kiểm tra bất biến I1: position_id phải bằng expected_pos_id
        i1_verdict = "PASS" if pos_id_val == expected_pos_id else "FAIL"
        print(f"[CHECK] step={step} id=I1 expected={expected_pos_id} actual={pos_id_val} verdict={i1_verdict}")
        if i1_verdict == "FAIL":
            all_i1_passed = False

        # Kiểm tra bất biến I2: mask_kv phải khớp phys_len (ở decode là expected_budget = 32)
        i2_verdict = "PASS" if mask_kv_len == phys_len else "FAIL"
        print(f"[CHECK] step={step} id=I2 expected={phys_len} actual={mask_kv_len} verdict={i2_verdict}")
        if i2_verdict == "FAIL":
            all_i2_passed = False

        if not all_i1_passed or not all_i2_passed:
            print(f"[RESULT] FAIL reason=invariant_violated_at_step_{step}")
            sys.exit(1)

    print(f"[CFG] position_id_seq={pos_id_sequence}")
    print("[RESULT] PASS reason=all_steps_verified")
    sys.exit(0)


if __name__ == "__main__":
    main()
