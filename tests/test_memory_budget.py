import unittest

import torch

from src.baselines.h2o_cache import H2OKVCache
from src.vector_cache.vector_cache import VectorKVCache


class TestMemoryBudget(unittest.TestCase):
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
