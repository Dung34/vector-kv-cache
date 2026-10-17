"""
CLI Runner: Needle-In-A-Haystack (NIAH) Benchmark (Full FP16 vs H2O Binary Eviction).
Usage:
    python scripts/run_niah_benchmark.py --model_id Qwen/Qwen2.5-1.5B-Instruct --context_lengths 2048 4096 8192
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
from src.evaluation.niah_evaluator import NIAHEvaluator
from src.utils.model_utils import load_model_and_tokenizer
from src.utils.plot_utils import plot_niah_heatmap


def main():
    parser = argparse.ArgumentParser(description="NIAH Retrieval Benchmark for Full vs H2O")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--context_lengths", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--depths", type=float, nargs="+", default=[0.0, 0.25, 0.50, 0.75, 1.0])
    parser.add_argument("--output_dir", type=str, default="results/niah_baseline")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting NIAH Benchmark on {device} ===")
    print(f"Model: {args.model_id}")
    print(f"Context lengths: {args.context_lengths}")
    print(f"Depths: {args.depths}")

    model, tokenizer = load_model_and_tokenizer(model_id=args.model_id, device=device)
    evaluator = NIAHEvaluator(model=model, tokenizer=tokenizer, max_new_tokens=20)

    configurations = [
        ("Full_FP16", FullKVCacheWrapper(), None),
        ("H2O_Compressed_75pct", H2OKVCache(compression_ratio=0.75), True),
    ]

    for exp_name, cache_inst, needs_hook in configurations:
        hook_mgr = None
        if needs_hook:
            hook_mgr = H2OAttentionHookManager(model=model, kv_cache=cache_inst)
            hook_mgr.register_hooks()

        try:
            report = evaluator.run_grid(
                context_lengths=args.context_lengths,
                depth_percents=args.depths,
                cache_manager=cache_inst,
                experiment_name=exp_name,
            )

            # Save JSON report
            json_file = os.path.join(args.output_dir, f"niah_{exp_name}.json")
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[Saved] Report to: {json_file}")

            # Plot Heatmap
            heatmap_file = os.path.join(args.output_dir, f"heatmap_{exp_name}.png")
            plot_niah_heatmap(
                matrix=report["matrix"],
                context_lengths=args.context_lengths,
                depth_percents=args.depths,
                title=f"NIAH Retrieval Accuracy: {exp_name} ({args.model_id.split('/')[-1]})",
                output_path=heatmap_file,
            )

        finally:
            if hook_mgr is not None:
                hook_mgr.remove_hooks()


if __name__ == "__main__":
    main()
