"""
Unit tests for VectorBudgetPartitioner.
Tests budget invariance, sink retention, and 3-tier partitioning.
"""

import unittest
import torch
from src.vector_cache.partitioner import VectorBudgetPartitioner


class TestVectorBudgetPartitioner(unittest.TestCase):
    def setUp(self):
        self.partitioner = VectorBudgetPartitioner(
            compression_ratio=0.75,
            approx_ratio=0.15,
            recent_ratio=0.10,
            min_recent_tokens=4,
            num_sink_tokens=4,
        )

    def test_compute_budgets_invariance(self):
        total_len = 200
        target_budget = 50  # 25% of 200
        k_sink, k_recent, n_r, n_k, n_a = self.partitioner.compute_budgets(total_len, target_budget)

        # Budget Swap Invariance: N_K + N_R == 2 * target_budget
        self.assertEqual(n_k + n_r, 2 * target_budget)
        # N_K > target_budget
        self.assertGreater(n_k, target_budget)
        # N_A == N_K - N_R
        self.assertEqual(n_a, n_k - n_r)
        # Sinks kept
        self.assertEqual(k_sink, 4)

    def test_partition_execution(self):
        batch_size = 1
        num_heads = 2
        total_len = 80
        head_dim = 16
        target_budget = 20

        k_full = torch.randn(batch_size, num_heads, total_len, head_dim)
        v_full = torch.randn(batch_size, num_heads, total_len, head_dim)
        pos_full = torch.arange(total_len).view(1, 1, -1).expand(batch_size, num_heads, -1)

        scores = torch.rand(batch_size, num_heads, total_len)
        cached_eps = torch.rand(batch_size, num_heads, total_len)

        res = self.partitioner.partition(
            layer_idx=0,
            k_full=k_full,
            v_full=v_full,
            pos_full=pos_full,
            target_budget=target_budget,
            accumulated_scores=scores,
            cached_eps=cached_eps,
        )

        n_k = res.k_cand.shape[-2]
        n_r = res.v_retain.shape[-2]
        self.assertEqual(n_k + n_r, 2 * target_budget)
        self.assertEqual(res.retain_rel.shape[-1], n_r)
        self.assertEqual(res.approx_rel.shape[-1], n_k - n_r)


if __name__ == "__main__":
    unittest.main()
