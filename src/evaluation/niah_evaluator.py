"""
Needle-In-A-Haystack (NIAH) Evaluator.
Executes grid evaluation across sequence lengths, insertion depths, and multi-needle scenarios.
Produces structured test metrics, recall statistics, and accuracy matrices.
"""

from typing import List, Dict, Any, Optional
import torch
from .niah_generator import NeedleInHaystackGenerator, MultiNeedleGenerator
from .memory_tracker import MemoryTracker, LatencyTimer


class NIAHEvaluator:
    """Runs Single-Needle and Multi-Needle NIAH benchmark across context lengths."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        generator: Optional[NeedleInHaystackGenerator] = None,
        multi_generator: Optional[MultiNeedleGenerator] = None,
        max_new_tokens: int = 20,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generator = generator or NeedleInHaystackGenerator()
        self.multi_generator = multi_generator or MultiNeedleGenerator()
        self.max_new_tokens = max_new_tokens
        self.device = next(model.parameters()).device

    # =========================================================================
    # SINGLE-NEEDLE EVALUATION
    # =========================================================================

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

        if hasattr(cache_manager, "reset_scores"):
            cache_manager.reset_scores()

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if cache_manager is not None:
            gen_kwargs["past_key_values"] = cache_manager

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        time_metrics = timer.stop(self.max_new_tokens)
        mem_metrics = memory_tracker.stop()

        new_tokens = outputs[0, actual_input_len:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
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
        """Runs Single-Needle grid evaluation."""
        results: List[Dict[str, Any]] = []
        matrix: Dict[int, Dict[float, int]] = {length: {} for length in context_lengths}

        print(f"\n--- Starting Single-Needle NIAH Grid: {experiment_name} ---")
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
                print(f"[{status}] Length={length:5d} | Depth={depth*100:5.1f}% | Output: '{res['response_text'][:40]}'")

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

    # =========================================================================
    # MULTI-NEEDLE EVALUATION (RULER / Multi-Key Needle Retrieval)
    # =========================================================================

    def evaluate_multi_needle_point(
        self,
        context_length: int,
        num_needles: int = 3,
        custom_depths: Optional[List[float]] = None,
        cache_manager: Optional[Any] = None,
        max_new_tokens: int = 80,
    ) -> Dict[str, Any]:
        """
        Runs Multi-Needle test at a specific context length.
        Checks how many needles among the scattered keys are successfully retrieved (Recall).
        """
        prompt, selected_needles = self.multi_generator.generate_prompt(
            tokenizer=self.tokenizer,
            target_token_length=context_length,
            num_needles=num_needles,
            custom_depths=custom_depths,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        actual_input_len = inputs.input_ids.shape[-1]

        memory_tracker = MemoryTracker(self.device)
        memory_tracker.start()
        timer = LatencyTimer()
        timer.start()

        if hasattr(cache_manager, "reset_scores"):
            cache_manager.reset_scores()

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if cache_manager is not None:
            gen_kwargs["past_key_values"] = cache_manager

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        time_metrics = timer.stop(max_new_tokens)
        mem_metrics = memory_tracker.stop()

        new_tokens = outputs[0, actual_input_len:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Score recall across all needles
        retrieved_needles = []
        for needle in selected_needles:
            if needle["passkey"].lower() in response_text.lower():
                retrieved_needles.append(needle["key"])

        retrieved_count = len(retrieved_needles)
        total_needles = len(selected_needles)
        recall = retrieved_count / total_needles if total_needles > 0 else 0.0
        all_retrieved = (retrieved_count == total_needles)

        return {
            "context_length": context_length,
            "actual_input_len": actual_input_len,
            "num_needles": total_needles,
            "retrieved_count": retrieved_count,
            "retrieved_keys": retrieved_needles,
            "recall": round(recall, 4),
            "all_retrieved": all_retrieved,
            "response_text": response_text,
            "memory": mem_metrics,
            "latency": time_metrics,
        }

    def run_multi_needle_grid(
        self,
        context_lengths: List[int],
        num_needles_list: List[int] = [3, 5],
        cache_manager: Optional[Any] = None,
        experiment_name: str = "baseline",
    ) -> Dict[str, Any]:
        """
        Runs Multi-Needle benchmark across context lengths and needle counts (3 and 5 needles).
        """
        results: List[Dict[str, Any]] = []
        recall_matrix: Dict[int, Dict[int, float]] = {length: {} for length in context_lengths}

        print(f"\n--- Starting Multi-Needle Benchmark: {experiment_name} ---")
        for length in context_lengths:
            for n_needles in num_needles_list:
                res = self.evaluate_multi_needle_point(
                    context_length=length,
                    num_needles=n_needles,
                    cache_manager=cache_manager,
                )
                recall_matrix[length][n_needles] = res["recall"]
                results.append(res)
                print(f"[Multi-Needle] Length={length:5d} | Needles={n_needles} | Recall: {res['recall']*100:5.1f}% ({res['retrieved_count']}/{n_needles})")

        avg_recall = sum(r["recall"] for r in results) / len(results) if results else 0.0
        all_passed_count = sum(1 for r in results if r["all_retrieved"])

        return {
            "experiment_name": experiment_name,
            "context_lengths": context_lengths,
            "num_needles_list": num_needles_list,
            "average_recall": round(avg_recall, 4),
            "all_passed_count": all_passed_count,
            "total_tests": len(results),
            "recall_matrix": recall_matrix,
            "detailed_results": results,
        }
