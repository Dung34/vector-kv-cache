"""
Unit tests for H2OKVCache and Attention Hook processing using mock tensors.
Runs on CPU without requiring GPU hardware or model downloads.
"""

import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baselines.h2o_cache import H2OKVCache
from src.baselines.h2o_hook import H2OAttentionHookManager


class TestH2OKVCache(unittest.TestCase):
    def test_h2o_budget_enforcement(self):
        batch_size = 1
        num_heads = 2
        seq_len = 200
        head_dim = 64
        compression_ratio = 0.50 # Keep 50% = 100 tokens

        cache = H2OKVCache(
            compression_ratio=compression_ratio,
            recent_ratio=0.20, # 20 tokens recent
            min_recent_tokens=20,
        )

        k_mock = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v_mock = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Mock attention scores: token 10 and 20 are given huge scores (Heavy Hitters)
        mock_scores = torch.zeros(batch_size, num_heads, seq_len)
        mock_scores[:, :, 10] = 1000.0
        mock_scores[:, :, 20] = 500.0

        cache.register_attention_scores(layer_idx=0, attn_scores=mock_scores)

        # Run update (prefill)
        k_out, v_out = cache.update(k_mock, v_mock, layer_idx=0)

        expected_budget = int((1.0 - compression_ratio) * seq_len)
        # Prefill preserves all tokens so causal attention mask in Hugging Face matches
        self.assertEqual(k_out.shape, (batch_size, num_heads, seq_len, head_dim))
        self.assertEqual(v_out.shape, (batch_size, num_heads, seq_len, head_dim))
        self.assertEqual(cache.get_seq_length(layer_idx=0), seq_len)

        # Test decoding step: append 1 token
        k_step = torch.randn(batch_size, num_heads, 1, head_dim)
        v_step = torch.randn(batch_size, num_heads, 1, head_dim)
        step_scores = torch.ones(batch_size, num_heads, 1)

        cache.register_attention_scores(layer_idx=0, attn_scores=step_scores)
        k_out2, v_out2 = cache.update(k_step, v_step, layer_idx=0)

        # Budget must remain capped at expected_budget
        self.assertEqual(k_out2.shape[-2], expected_budget)
        self.assertEqual(v_out2.shape[-2], expected_budget)

    def test_gqa_head_processing(self):
        # Test Qwen2.5 GQA setup: 12 Query heads -> 2 KV heads (ratio 6:1)
        class DummyModel:
            pass

        cache = H2OKVCache()
        hook_mgr = H2OAttentionHookManager(
            model=DummyModel(),
            kv_cache=cache,
            num_query_heads=12,
            num_kv_heads=2,
        )

        batch_size = 2
        query_len = 1
        seq_len = 32
        # Mock attention weights: [batch, 12, 1, 32]
        attn_weights = torch.ones(batch_size, 12, query_len, seq_len)

        processed = hook_mgr._process_gqa_attention_scores(attn_weights)
        # Output must be [batch, 2, 1, 32]
        self.assertEqual(processed.shape, (batch_size, 2, query_len, seq_len))
        self.assertTrue(torch.allclose(processed, torch.ones(batch_size, 2, query_len, seq_len)))

    def test_attention_sinks_retention(self):
        """
        Verifies that initial tokens (num_sink_tokens) are never evicted,
        even if their attention scores are zero.
        """
        batch_size = 1
        num_heads = 2
        seq_len = 100
        head_dim = 32
        num_sinks = 4

        cache = H2OKVCache(
            compression_ratio=0.75,
            recent_ratio=0.10,
            min_recent_tokens=8,
            num_sink_tokens=num_sinks,
        )

        k_mock = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v_mock = torch.randn(batch_size, num_heads, seq_len, head_dim)
        # Mark sink tokens with a specific identifier value
        k_mock[:, :, :num_sinks, :] = 999.0

        # Scores: zero on sink tokens, high on middle tokens
        scores = torch.zeros(batch_size, num_heads, seq_len)
        scores[:, :, 10:20] = 50.0
        cache.register_attention_scores(0, scores)

        # Prefill
        cache.update(k_mock, v_mock, layer_idx=0)

        # Trigger prune with 1 decode token
        k_step = torch.randn(batch_size, num_heads, 1, head_dim)
        v_step = torch.randn(batch_size, num_heads, 1, head_dim)
        k_out, _ = cache.update(k_step, v_step, layer_idx=0)

        # First num_sinks tokens in k_out must match the marked sink tokens
        sink_keys = k_out[:, :, :num_sinks, :]
        self.assertTrue(
            torch.allclose(sink_keys, torch.tensor(999.0, device=sink_keys.device)),
            "Initial sink tokens must be permanently retained regardless of scores!"
        )


if __name__ == "__main__":
    unittest.main()
