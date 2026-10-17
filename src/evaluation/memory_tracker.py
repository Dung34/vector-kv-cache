"""
Memory and Latency Profiling Utilities for LLM / SLM KV Cache Benchmarking.
Measures Peak VRAM, Allocated Memory, TTFT (Time To First Token), and Generation Throughput.
"""

import time
import torch
from typing import Dict, Any, Optional


class MemoryTracker:
    """Tracks GPU VRAM consumption during inference (with graceful CPU fallback)."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.is_cuda = self.device.type == "cuda"
        self.initial_allocated: float = 0.0
        self.peak_allocated: float = 0.0

    def start(self):
        """Reset peak memory counter and record baseline allocated memory."""
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            self.initial_allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # MB
        else:
            self.initial_allocated = 0.0

    def stop(self) -> Dict[str, float]:
        """Stop tracking and return allocated and peak memory in Megabytes."""
        if self.is_cuda:
            current_allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            self.peak_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            kv_overhead = max(0.0, current_allocated - self.initial_allocated)
            return {
                "initial_allocated_mb": round(self.initial_allocated, 2),
                "current_allocated_mb": round(current_allocated, 2),
                "peak_allocated_mb": round(self.peak_allocated, 2),
                "delta_allocated_mb": round(kv_overhead, 2),
            }
        else:
            return {
                "initial_allocated_mb": 0.0,
                "current_allocated_mb": 0.0,
                "peak_allocated_mb": 0.0,
                "delta_allocated_mb": 0.0,
            }


class LatencyTimer:
    """Measures Prefill Latency, TTFT (Time to First Token), and Generation Throughput."""

    def __init__(self):
        self.t_start: float = 0.0
        self.t_first_token: float = 0.0
        self.t_end: float = 0.0
        self.num_generated_tokens: int = 0

    def start(self):
        self.t_start = time.perf_counter()
        self.t_first_token = 0.0
        self.t_end = 0.0
        self.num_generated_tokens = 0

    def mark_first_token(self):
        if self.t_first_token == 0.0:
            self.t_first_token = time.perf_counter()

    def stop(self, num_generated_tokens: int) -> Dict[str, float]:
        self.t_end = time.perf_counter()
        self.num_generated_tokens = num_generated_tokens

        total_time_s = self.t_end - self.t_start
        ttft_s = (self.t_first_token - self.t_start) if self.t_first_token > 0.0 else total_time_s
        decode_time_s = max(1e-6, total_time_s - ttft_s)
        throughput = (num_generated_tokens - 1) / decode_time_s if num_generated_tokens > 1 else num_generated_tokens / total_time_s

        return {
            "total_time_sec": round(total_time_s, 4),
            "ttft_ms": round(ttft_s * 1000, 2),
            "throughput_tokens_per_sec": round(throughput, 2),
            "num_generated_tokens": num_generated_tokens,
        }
