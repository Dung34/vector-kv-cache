"""
Unit tests for FullKVCacheWrapper using mock tensors (CPU compatible).
"""

import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baselines.full_cache import FullKVCacheWrapper


class TestFullKVCache(unittest.TestCase):
    def test_full_cache_initialization_and_update(self):
        cache = FullKVCacheWrapper()
        self.assertEqual(cache.name, "Full-FP16-Cache")
        self.assertEqual(cache.compression_ratio, 0.0)

        batch_size = 2
        num_heads = 2
        seq_len = 64
        head_dim = 128

        # Prefill step
        k_init = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v_init = torch.randn(batch_size, num_heads, seq_len, head_dim)

        k_out, v_out = cache.update(k_init, v_init, layer_idx=0)
        self.assertEqual(k_out.shape, (batch_size, num_heads, seq_len, head_dim))
        self.assertEqual(v_out.shape, (batch_size, num_heads, seq_len, head_dim))
        self.assertEqual(cache.get_seq_length(layer_idx=0), seq_len)

        # Autoregressive decoding step (seq_len = 1)
        k_step = torch.randn(batch_size, num_heads, 1, head_dim)
        v_step = torch.randn(batch_size, num_heads, 1, head_dim)

        k_out_step, v_out_step = cache.update(k_step, v_step, layer_idx=0)
        self.assertEqual(k_out_step.shape, (batch_size, num_heads, seq_len + 1, head_dim))
        self.assertEqual(v_out_step.shape, (batch_size, num_heads, seq_len + 1, head_dim))
        self.assertEqual(cache.get_seq_length(layer_idx=0), seq_len + 1)

        # Check memory bytes calculation
        mem_bytes = cache.get_memory_bytes()
        expected_bytes = 2 * (batch_size * num_heads * (seq_len + 1) * head_dim * k_init.element_size())
        self.assertEqual(mem_bytes, expected_bytes)


if __name__ == "__main__":
    unittest.main()
