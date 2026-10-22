import unittest

import torch

from src.baselines.h2o_cache import H2OKVCache
from src.vector_cache.vector_cache import VectorKVCache


class TestLogicalPhysicalLengths(unittest.TestCase):
    def _run_prefill_and_decode(self, cache):
        prefill = torch.zeros(1, 2, 8, 4)
        token = torch.zeros(1, 2, 1, 4)
        cache.update(prefill, prefill, layer_idx=0)
        cache.update(token, token, layer_idx=0)
        return cache._key_cache[0].shape[-2]

    def test_h2o_keeps_logical_length_after_prune(self):
        cache = H2OKVCache(compression_ratio=0.5, min_recent_tokens=2)
        physical_len = self._run_prefill_and_decode(cache)
        self.assertEqual(cache.get_seq_length(0), 9)
        self.assertLessEqual(physical_len, 9)

    def test_vector_keeps_logical_length_after_prune(self):
        cache = VectorKVCache(
            compression_ratio=0.5,
            approx_ratio=0.1,
            min_recent_tokens=2,
            w_ols=torch.eye(4).unsqueeze(0).unsqueeze(0),
        )
        physical_len = self._run_prefill_and_decode(cache)
        self.assertEqual(cache.get_seq_length(0), 9)
        self.assertLessEqual(physical_len, 9)

    def test_vector_mask_length_uses_physical_budget_after_prune(self):
        cache = VectorKVCache(
            compression_ratio=0.5,
            approx_ratio=0.1,
            min_recent_tokens=2,
            w_ols=torch.eye(4).unsqueeze(0).unsqueeze(0),
        )
        physical_len = self._run_prefill_and_decode(cache)
        mask_len, offset = cache.get_mask_sizes(torch.tensor([8]), layer_idx=0)
        self.assertEqual(mask_len, physical_len)
        self.assertEqual(offset, 0)


if __name__ == "__main__":
    unittest.main()
