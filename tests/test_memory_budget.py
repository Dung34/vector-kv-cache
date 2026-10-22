import unittest

import torch

from src.baselines.h2o_cache import H2OKVCache
from src.vector_cache.vector_cache import VectorKVCache


class TestMemoryBudget(unittest.TestCase):
    def test_h2o_prune_keeps_scores_aligned_without_empty_cache(self):
        original_empty_cache = torch.cuda.empty_cache
        torch.cuda.empty_cache = lambda: (_ for _ in ()).throw(AssertionError("empty_cache is not part of cache update"))
        try:
            cache = H2OKVCache(target_budget=16, max_physical_budget=16, min_recent_tokens=4)
            prefill = torch.randn(1, 2, 32, 4)
            cache.update(prefill, prefill, layer_idx=0)
            cache.update(torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4), layer_idx=0)
            self.assertEqual(cache.accumulated_scores[0].shape[-1], cache._physical_len(0))
        finally:
            torch.cuda.empty_cache = original_empty_cache

    def test_h2o_caps_prefill_physical_length(self):
        cache = H2OKVCache(
            compression_ratio=0.5,
            target_budget=16,
            max_physical_budget=16,
            min_recent_tokens=4,
        )
        states = torch.zeros(1, 2, 64, 4)
        cache.update(states, states, layer_idx=0)
        self.assertLessEqual(cache._physical_len(0), 16)

    def test_h2o_keeps_cap_across_prefill_chunks(self):
        cache = H2OKVCache(target_budget=16, max_physical_budget=16, min_recent_tokens=4)
        chunk = torch.zeros(1, 2, 32, 4)
        cache.update(chunk, chunk, layer_idx=0)
        cache.update(chunk, chunk, layer_idx=0)
        self.assertLessEqual(cache._physical_len(0), 16)

    def test_h2o_rejects_target_above_physical_cap(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            H2OKVCache(target_budget=17, max_physical_budget=16)

    def test_vector_accepts_target_below_physical_cap(self):
        cache = VectorKVCache(
            compression_ratio=0.5,
            approx_ratio=0.1,
            target_budget=16,
            max_physical_budget=32,
            min_recent_tokens=4,
            w_ols=torch.eye(4).unsqueeze(0).unsqueeze(0),
        )
        states = torch.zeros(1, 2, 64, 4)
        cache.update(states, states, layer_idx=0)
        self.assertLessEqual(cache._physical_len(0), 32)

    def test_vector_keeps_cap_across_prefill_chunks(self):
        cache = VectorKVCache(
            compression_ratio=0.5,
            approx_ratio=0.1,
            target_budget=16,
            max_physical_budget=32,
            min_recent_tokens=4,
            w_ols=torch.eye(4).unsqueeze(0).unsqueeze(0),
        )
        chunk = torch.zeros(1, 2, 32, 4)
        cache.update(chunk, chunk, layer_idx=0)
        cache.update(chunk, chunk, layer_idx=0)
        self.assertLessEqual(cache._physical_len(0), 32)

    def test_vector_prune_keeps_auxiliary_state_aligned_without_empty_cache(self):
        original_empty_cache = torch.cuda.empty_cache
        torch.cuda.empty_cache = lambda: (_ for _ in ()).throw(AssertionError("empty_cache is not part of cache update"))
        try:
            cache = VectorKVCache(
                compression_ratio=0.5,
                approx_ratio=0.1,
                target_budget=16,
                max_physical_budget=32,
                min_recent_tokens=4,
                w_ols=torch.eye(4).unsqueeze(0).unsqueeze(0),
            )
            prefill = torch.randn(1, 2, 32, 4)
            cache.update(prefill, prefill, layer_idx=0)
            cache.update(torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4), layer_idx=0)
            physical = cache._physical_len(0)
            self.assertEqual(cache._stored_positions[0].shape[-1], physical)
            self.assertEqual(cache._cached_eps[0].shape[-1], physical)
            self.assertEqual(cache.accumulated_scores[0].shape[-1], physical)
            self.assertEqual(
                cache._retain_indices[0].shape[-1] + cache._approx_indices[0].shape[-1],
                physical,
            )
        finally:
            torch.cuda.empty_cache = original_empty_cache

    def test_vector_rejects_incompatible_budgets(self):
        with self.assertRaisesRegex(ValueError, "requires more physical keys"):
            VectorKVCache(
                compression_ratio=0.5,
                approx_ratio=0.1,
                target_budget=32,
                max_physical_budget=32,
                w_ols=torch.eye(4).unsqueeze(0).unsqueeze(0),
            )


if __name__ == "__main__":
    unittest.main()
