"""
CLI Runner: Memory and Throughput Benchmark (Full FP16 vs H2O Binary Eviction).
Usage:
    python scripts/run_memory_benchmark.py --model_id Qwen/Qwen2.5-1.5B-Instruct --seq_len 4096
"""

import argparse
import os
import sys
import json
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baselines.full_cache import FullKVCacheWrapper
from src.baselines.h2o_cache import H2OKVCache
from src.baselines.h2o_hook import H2OAttentionHookManager
from src.evaluation.memory_tracker import MemoryTracker, LatencyTimer
from src.utils.model_utils import load_model_and_tokenizer
from src.utils.plot_utils import plot_vram_comparison


def run_single_memory_test(model, tokenizer, prompt: str, cache_obj, hook_mgr=None, max_new_tokens=128):
    import time
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[-1]

    if hasattr(cache_obj, "reset_scores"):
        cache_obj.reset_scores()

    tracker = MemoryTracker(device)
    tracker.start()

    # 1. Prefill phase measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t_prefill_0 = time.perf_counter()

    prefill_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "use_cache": True,
    }
    if cache_obj is not None:
        prefill_kwargs["past_key_values"] = cache_obj

    with torch.no_grad():
        prefill_out = model(**prefill_kwargs)

    first_token_logits = prefill_out.logits[:, -1, :]
    next_token = torch.argmax(first_token_logits, dim=-1, keepdim=True)
    active_past = prefill_out.past_key_values if cache_obj is None else cache_obj

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t_prefill_1 = time.perf_counter()
    prefill_ms = (t_prefill_1 - t_prefill_0) * 1000.0

    # 2. Decode phase measurement
    curr_token = next_token
    decode_steps = max(1, max_new_tokens - 1)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t_decode_0 = time.perf_counter()

    with torch.no_grad():
        for _ in range(decode_steps):
            step_kwargs = {
                "input_ids": curr_token,
                "use_cache": True,
                "past_key_values": active_past,
            }
            step_out = model(**step_kwargs)
            curr_token = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
            if cache_obj is None:
                active_past = step_out.past_key_values

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t_decode_1 = time.perf_counter()

    decode_duration_s = t_decode_1 - t_decode_0
    decode_tok_s = decode_steps / max(1e-5, decode_duration_s)
    itl_ms = (decode_duration_s / decode_steps) * 1000.0

    mem_stats = tracker.stop()

    return {
        "input_len": input_len,
        "generated_tokens": max_new_tokens,
        "memory": mem_stats,
        "latency": {
            "prefill_ms": round(prefill_ms, 2),
            "decode_tok_s": round(decode_tok_s, 2),
            "itl_ms": round(itl_ms, 2),
            "decode_duration_s": round(decode_duration_s, 4),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Memory & Throughput Benchmark for Full vs H2O")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--new_tokens", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="results/memory_baseline")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting Memory Benchmark on {device} ===")
    print(f"Model: {args.model_id} | Context: {args.seq_len} tokens | New Tokens: {args.new_tokens}")

    model, tokenizer = load_model_and_tokenizer(model_id=args.model_id, device=device)

    # Generate synthetic dummy text of length args.seq_len
    sample_text = "The quick brown fox jumps over the lazy dog. Artificial intelligence revolutionizes science. "
    tokens_per_repeat = len(tokenizer.encode(sample_text, add_special_tokens=False))
    repeats = (args.seq_len // tokens_per_repeat) + 1
    long_prompt = (sample_text * repeats)[: args.seq_len * 4]
    # Trim to exact tokens
    input_ids = tokenizer.encode(long_prompt, add_special_tokens=False)[: args.seq_len]
    prompt = tokenizer.decode(input_ids)

    configurations = [
        ("Full-FP16 (p_c=0%)", FullKVCacheWrapper(), None),
        ("H2O (p_c=50%)", H2OKVCache(compression_ratio=0.50), True),
        ("H2O (p_c=75%)", H2OKVCache(compression_ratio=0.75), True),
        ("H2O (p_c=85%)", H2OKVCache(compression_ratio=0.85), True),
    ]

    all_results = {}
    vram_plot_data = {}

    for name, cache_inst, needs_hook in configurations:
        print(f"\nEvaluating: {name}...")
        hook_mgr = None
        if needs_hook:
            hook_mgr = H2OAttentionHookManager(model=model, kv_cache=cache_inst)
            hook_mgr.register_hooks()

        try:
            stats = run_single_memory_test(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                cache_obj=cache_inst,
                hook_mgr=hook_mgr,
                max_new_tokens=args.new_tokens,
            )
            all_results[name] = stats
            peak_mb = stats["memory"]["peak_allocated_mb"]
            vram_plot_data[name] = peak_mb
            print(
                f"-> Peak VRAM: {peak_mb:.1f} MB | "
                f"Prefill: {stats['latency']['prefill_ms']:.1f} ms | "
                f"Decode: {stats['latency']['decode_tok_s']:.1f} tok/s ({stats['latency']['itl_ms']:.1f} ms/tok)"
            )
        finally:
            if hook_mgr is not None:
                hook_mgr.remove_hooks()

    # Save JSON report
    json_path = os.path.join(args.output_dir, f"memory_results_{args.seq_len}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved] Detailed JSON metrics to: {json_path}")

    # Generate Chart
    chart_path = os.path.join(args.output_dir, f"vram_comparison_{args.seq_len}.png")
    plot_vram_comparison(
        vram_plot_data,
        title=f"Peak VRAM Comparison @ {args.seq_len} Tokens ({args.model_id.split('/')[-1]})",
        output_path=chart_path,
    )


if __name__ == "__main__":
    main()
