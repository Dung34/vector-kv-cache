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

    def test_cached_index_buffers_and_monotonicity(self):
        """
        Verifies that cached index buffers are reused across decode steps,
        and that pruning keeps the cache monotonic without full sorting overhead.
        """
        batch_size = 1
        num_heads = 2
        seq_len = 80
        head_dim = 16
        cache = H2OKVCache(compression_ratio=0.5, num_sink_tokens=4, min_recent_tokens=8)

        # Prefill: buffers should not be allocated yet
        self.assertIsNone(cache._cached_sink_indices)
        self.assertIsNone(cache._cached_recent_offsets)

        k_mock = torch.arange(seq_len, dtype=torch.float32).view(1, 1, seq_len, 1).expand(batch_size, num_heads, seq_len, head_dim)
        v_mock = torch.randn(batch_size, num_heads, seq_len, head_dim)
        cache.update(k_mock, v_mock, layer_idx=0)

        # Step 1: Decode token
        k_step = torch.full((batch_size, num_heads, 1, head_dim), float(seq_len), dtype=torch.float32)
        v_step = torch.randn(batch_size, num_heads, 1, head_dim)
        k_out1, _ = cache.update(k_step, v_step, layer_idx=0)

        # Check buffers are created
        self.assertIsNotNone(cache._cached_sink_indices)
        self.assertIsNotNone(cache._cached_recent_offsets)
        sink_buf_ref = cache._cached_sink_indices
        recent_buf_ref = cache._cached_recent_offsets

        # Step 2: Next decode token
        k_step2 = torch.full((batch_size, num_heads, 1, head_dim), float(seq_len + 1), dtype=torch.float32)
        v_step2 = torch.randn(batch_size, num_heads, 1, head_dim)
        k_out2, _ = cache.update(k_step2, v_step2, layer_idx=0)

        # Buffers must be the exact same objects in memory (reused!)
        self.assertIs(cache._cached_sink_indices, sink_buf_ref)
        self.assertIs(cache._cached_recent_offsets, recent_buf_ref)

        # Token positions in k_out2 must be strictly monotonically increasing
        positions = k_out2[0, 0, :, 0]
        self.assertTrue(torch.all(positions[1:] > positions[:-1]), "KV cache must remain strictly sorted!")

    def test_query_chunk_attention_injection_preserves_target_tokens(self):
        """
        Verifies that Query Chunk Attention Injection preserves target tokens in H2OKVCache.
        """
        batch_size = 1
        num_heads = 2
        head_dim = 16
        cache = H2OKVCache(compression_ratio=0.75, num_sink_tokens=4, min_recent_tokens=8)

        haystack_len = 90
        query_len = 10
        total_prefill = haystack_len + query_len

        # Haystack
        cache.is_prefilling = True
        k_haystack = torch.randn(batch_size, num_heads, haystack_len, head_dim)
        v_haystack = torch.randn(batch_size, num_heads, haystack_len, head_dim)
        cache.update(k_haystack, v_haystack, layer_idx=0)

        # Query chunk
        k_query = torch.randn(batch_size, num_heads, query_len, head_dim)
        v_query = torch.randn(batch_size, num_heads, query_len, head_dim)
        cache.update(k_query, v_query, layer_idx=0)

        # Register query attention
        target_token_idx = 50
        mock_query_attn = torch.zeros(batch_size, num_heads, query_len, total_prefill)
        mock_query_attn[:, :, :, target_token_idx] = 10.0
        cache.register_attention_scores(layer_idx=0, attn_scores=mock_query_attn)

        # Trigger prune
        cache.is_prefilling = False
        k_dec = torch.randn(batch_size, num_heads, 1, head_dim)
        v_dec = torch.randn(batch_size, num_heads, 1, head_dim)
        cache.update(k_dec, v_dec, layer_idx=0)

        stored_pos = cache._stored_positions[0]
        self.assertIsNotNone(stored_pos)
        stored_list = stored_pos[0, 0].cpu().tolist()
        self.assertIn(target_token_idx, stored_list)


if __name__ == "__main__":
    unittest.main()

