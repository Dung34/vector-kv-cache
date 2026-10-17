"""
Needle-In-A-Haystack (NIAH) Evaluator.
Executes grid evaluation across sequence lengths and insertion depths,
producing structured test metrics and accuracy matrices for heatmap plotting.
"""

from typing import List, Dict, Any, Optional
import torch
from .niah_generator import NeedleInHaystackGenerator
from .memory_tracker import MemoryTracker, LatencyTimer


class NIAHEvaluator:
    """Runs NIAH benchmark grid across context lengths and needle depths."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        generator: Optional[NeedleInHaystackGenerator] = None,
        max_new_tokens: int = 20,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generator = generator or NeedleInHaystackGenerator()
        self.max_new_tokens = max_new_tokens
        self.device = next(model.parameters()).device

    def evaluate_point(
        self,
        context_length: int,
        depth_percent: float,
        cache_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Runs a single NIAH test point."""
        prompt, expected_answer = self.generator.generate_prompt(
            tokenizer=self.tokenizer,
            target_token_length=context_length,
            depth_percent=depth_percent,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        actual_input_len = inputs.input_ids.shape[-1]

        memory_tracker = MemoryTracker(self.device)
        memory_tracker.start()
        timer = LatencyTimer()
        timer.start()

        # If cache_manager has reset_scores, reset for fresh sequence
        if hasattr(cache_manager, "reset_scores"):
            cache_manager.reset_scores()

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False, # greedy decoding for deterministic evaluation
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if cache_manager is not None:
            gen_kwargs["past_key_values"] = cache_manager

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        time_metrics = timer.stop(self.max_new_tokens)
        mem_metrics = memory_tracker.stop()

        # Extract generated tokens
        new_tokens = outputs[0, actual_input_len:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Check retrieval success
        is_correct = expected_answer.lower() in response_text.lower()

        return {
            "context_length": context_length,
            "actual_input_len": actual_input_len,
            "depth_percent": depth_percent,
            "expected_answer": expected_answer,
            "response_text": response_text,
            "is_correct": bool(is_correct),
            "memory": mem_metrics,
            "latency": time_metrics,
        }

    def run_grid(
        self,
        context_lengths: List[int],
        depth_percents: List[float],
        cache_manager: Optional[Any] = None,
        experiment_name: str = "baseline",
    ) -> Dict[str, Any]:
        """
        Runs evaluation over all combinations of context_lengths and depth_percents.
        """
        results: List[Dict[str, Any]] = []
        matrix: Dict[int, Dict[float, int]] = {length: {} for length in context_lengths}

        print(f"\n--- Starting NIAH Grid Benchmark: {experiment_name} ---")
        for length in context_lengths:
            for depth in depth_percents:
                res = self.evaluate_point(
                    context_length=length,
                    depth_percent=depth,
                    cache_manager=cache_manager,
                )
                matrix[length][depth] = 1 if res["is_correct"] else 0
                results.append(res)
                status = "PASS" if res["is_correct"] else "FAIL"
                print(f"[{status}] Length={length:5d} | Depth={depth*100:5.1f}% | Response: '{res['response_text'][:40]}'")

        correct_count = sum(r["is_correct"] for r in results)
        total_count = len(results)
        overall_accuracy = (correct_count / total_count) if total_count > 0 else 0.0

        return {
            "experiment_name": experiment_name,
            "context_lengths": context_lengths,
            "depth_percents": depth_percents,
            "overall_accuracy": round(overall_accuracy, 4),
            "matrix": matrix,
            "detailed_results": results,
        }
