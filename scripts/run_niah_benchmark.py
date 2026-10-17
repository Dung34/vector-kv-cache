"""
CLI Runner: Needle-In-A-Haystack (Single-Needle & Multi-Needle) Benchmark.
Usage:
    python scripts/run_niah_benchmark.py --model_id Qwen/Qwen2.5-1.5B-Instruct --context_lengths 2048 4096 8192 --eval_type both
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
from src.utils.plot_utils import plot_niah_heatmap, plot_multi_needle_comparison


def main():
    parser = argparse.ArgumentParser(description="NIAH Retrieval Benchmark (Single & Multi-Needle)")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--context_lengths", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--depths", type=float, nargs="+", default=[0.0, 0.25, 0.50, 0.75, 1.0])
    parser.add_argument("--eval_type", type=str, choices=["single", "multi", "both"], default="both")
    parser.add_argument("--num_needles", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--output_dir", type=str, default="results/niah_baseline")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting NIAH Benchmark ({args.eval_type.upper()}) on {device} ===")
    print(f"Model: {args.model_id} | Context lengths: {args.context_lengths}")

    model, tokenizer = load_model_and_tokenizer(model_id=args.model_id, device=device)
    evaluator = NIAHEvaluator(model=model, tokenizer=tokenizer)

    configurations = [
        ("Full_FP16", FullKVCacheWrapper(), None),
        ("H2O_50pct", H2OKVCache(compression_ratio=0.50), True),
        ("H2O_75pct", H2OKVCache(compression_ratio=0.75), True),
    ]

    multi_needle_summary = {}

    for exp_name, cache_inst, needs_hook in configurations:
        hook_mgr = None
        if needs_hook:
            hook_mgr = H2OAttentionHookManager(model=model, kv_cache=cache_inst)
            hook_mgr.register_hooks()

        try:
            # 1. Single-Needle Evaluation
            if args.eval_type in ["single", "both"]:
                report_single = evaluator.run_grid(
                    context_lengths=args.context_lengths,
                    depth_percents=args.depths,
                    cache_manager=cache_inst,
                    experiment_name=exp_name,
                )
                json_file = os.path.join(args.output_dir, f"niah_single_{exp_name}.json")
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(report_single, f, indent=2)

                heatmap_file = os.path.join(args.output_dir, f"heatmap_{exp_name}.png")
                plot_niah_heatmap(
                    matrix=report_single["matrix"],
                    context_lengths=args.context_lengths,
                    depth_percents=args.depths,
                    title=f"Single-Needle Accuracy: {exp_name} ({args.model_id.split('/')[-1]})",
                    output_path=heatmap_file,
                )

            # 2. Multi-Needle Evaluation
            if args.eval_type in ["multi", "both"]:
                report_multi = evaluator.run_multi_needle_grid(
                    context_lengths=args.context_lengths,
                    num_needles_list=args.num_needles,
                    cache_manager=cache_inst,
                    experiment_name=exp_name,
                )
                json_multi_file = os.path.join(args.output_dir, f"niah_multi_{exp_name}.json")
                with open(json_multi_file, "w", encoding="utf-8") as f:
                    json.dump(report_multi, f, indent=2)

                # Store 4096 context results for summary chart
                target_len = 4096 if 4096 in args.context_lengths else args.context_lengths[0]
                multi_needle_summary[exp_name] = report_multi["recall_matrix"].get(target_len, {})

        finally:
            if hook_mgr is not None:
                hook_mgr.remove_hooks()

    # Generate Multi-Needle Comparative Chart
    if args.eval_type in ["multi", "both"] and multi_needle_summary:
        target_len = 4096 if 4096 in args.context_lengths else args.context_lengths[0]
        multi_chart_file = os.path.join(args.output_dir, f"multi_needle_recall_{target_len}.png")
        plot_multi_needle_comparison(
            multi_needle_summary,
            context_length=target_len,
            title="Multi-Needle Retrieval Recall (Full vs H2O)",
            output_path=multi_chart_file,
        )
        print(f"\n[Saved] Multi-Needle comparative chart to: {multi_chart_file}")


if __name__ == "__main__":
    main()
