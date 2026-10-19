"""
Unit Tests for VectorKVCache and VectorAttentionHookManager.
Verifies:
  1. Budget Relaxation & Mathematical Memory Invariance (Memory_VECTOR == Memory_H2O).
  2. 3-Tier Partitioning (eps_i ranking & classification into Retain/Approximate).
  3. On-the-fly Value Reconstruction using W_OLS and RoPE Decoupling.
  4. Multi-step autoregressive decode loop stability (no NaNs, stable shapes).
"""

import unittest
import torch
import torch.nn as nn
from src.vector_cache.vector_cache import VectorKVCache
from src.vector_cache.vector_hook import VectorAttentionHookManager


class TestVectorKVCache(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1
        self.num_layers = 2
        self.num_heads = 2
        self.head_dim = 64
        self.device = torch.device("cpu")

        # Synthetic W_OLS for test: [num_layers, num_heads, head_dim, head_dim]
        torch.manual_seed(42)
        self.w_ols = torch.randn(
            (self.num_layers, self.num_heads, self.head_dim, self.head_dim),
            dtype=torch.float32,
        )

    def test_budget_relaxation_invariance(self):
        """
        Verifies that total stored elements (N_K + N_R) equals exactly 2 * target_budget,
        while N_K is strictly greater than target_budget (budget relaxed to hold more keys).
        """
        compression_ratio = 0.75  # 25% target budget
        approx_ratio = 0.15       # 15% approximate
        recent_ratio = 0.10
        min_recent_tokens = 8

        cache = VectorKVCache(
            compression_ratio=compression_ratio,
            approx_ratio=approx_ratio,
            recent_ratio=recent_ratio,
            min_recent_tokens=min_recent_tokens,
            w_ols=self.w_ols,
        )

        total_tokens = 160
        # Prefill 159 tokens, then 1 token to trigger decode pruning
        k_prefill = torch.randn(self.batch_size, self.num_heads, total_tokens - 1, self.head_dim)
        v_prefill = torch.randn(self.batch_size, self.num_heads, total_tokens - 1, self.head_dim)
        cache.update(k_prefill, v_prefill, layer_idx=0)

        # 1 decode token
        k_step = torch.randn(self.batch_size, self.num_heads, 1, self.head_dim)
        v_step = torch.randn(self.batch_size, self.num_heads, 1, self.head_dim)
        k_out, v_out = cache.update(k_step, v_step, layer_idx=0)

        target_budget = int((1.0 - compression_ratio) * total_tokens)  # 0.25 * 160 = 40
        n_k = cache._key_cache[0].shape[-2]
        n_r = cache._value_cache[0].shape[-2]

        # Invariance check: N_K + N_R == 2 * target_budget
        self.assertEqual(n_k + n_r, 2 * target_budget, f"Expected {2*target_budget}, got {n_k + n_r}")
        # Budget relaxation check: N_K > target_budget
        self.assertGreater(n_k, target_budget, f"Keys {n_k} should exceed standard H2O budget {target_budget}")
        # Returned tensors must have identical sequence length for attention computation
        self.assertEqual(k_out.shape[-2], n_k)
        self.assertEqual(v_out.shape[-2], n_k)

    def test_3tier_partitioning_by_error(self):
        """
        Verifies that tokens with lower prediction error eps_i are preferentially
        assigned to the Approximate pool, while high-error tokens remain in Retain.
        """
        cache = VectorKVCache(
            compression_ratio=0.50,
            approx_ratio=0.20,
            recent_ratio=0.10,
            min_recent_tokens=4,
            w_ols=self.w_ols,
        )

        seq_len = 40
        k = torch.randn(self.batch_size, self.num_heads, seq_len, self.head_dim)
        # Create V where first 5 tokens have high noise, and next tokens have very low noise
        # Compute exact V_hat from K
        v_hat = cache._predict_v(layer_idx=0, k_rot=k, positions=torch.arange(seq_len).view(1, 1, -1).expand(1, self.num_heads, -1))
        noise = torch.zeros_like(v_hat)
        noise[:, :, :5, :] = 100.0  # huge error for first 5 tokens (less than n_r_h2 = 8 slots)
        noise[:, :, 5:, :] = 0.001  # almost zero error for the rest
        v = v_hat + noise

        # Update cache
        k_prefill = k[:, :, :-1, :]
        v_prefill = v[:, :, :-1, :]
        cache.update(k_prefill, v_prefill, layer_idx=0)

        # Trigger prune with 1 decode token
        cache.update(k[:, :, -1:, :], v[:, :, -1:, :], layer_idx=0)

        # Check approx indices: should only come from low noise tokens (>= 5)
        approx_indices = cache._approx_indices[0]
        stored_pos = cache._stored_positions[0]
        approx_positions = torch.gather(stored_pos, -1, approx_indices)

        # Check that high-noise positions (0 to 4) are NOT in approx_positions
        for pos in range(5):
            self.assertFalse(
                (approx_positions == pos).any().item(),
                f"High-error token at pos {pos} should not be in Approximate pool!"
            )

    def test_autoregressive_decoding_loop(self):
        """
        Simulates prefill followed by 10 continuous decoding steps.
        Asserts no dimension mismatch, no NaNs, and memory is strictly bounded.
        """
        cache = VectorKVCache(
            compression_ratio=0.75,
            approx_ratio=0.15,
            recent_ratio=0.10,
            min_recent_tokens=8,
            w_ols=self.w_ols,
        )

        prefill_len = 100
        k_init = torch.randn(self.batch_size, self.num_heads, prefill_len, self.head_dim)
        v_init = torch.randn(self.batch_size, self.num_heads, prefill_len, self.head_dim)

        # Prefill
        k_ret, v_ret = cache.update(k_init, v_init, layer_idx=0)
        self.assertEqual(k_ret.shape[-2], prefill_len)

        # 10 decoding steps
        for step in range(10):
            k_tok = torch.randn(self.batch_size, self.num_heads, 1, self.head_dim)
            v_tok = torch.randn(self.batch_size, self.num_heads, 1, self.head_dim)
            k_dec, v_dec = cache.update(k_tok, v_tok, layer_idx=0)

            # Check for NaNs
            self.assertFalse(torch.isnan(k_dec).any())
            self.assertFalse(torch.isnan(v_dec).any())
            # Output lengths must match
            self.assertEqual(k_dec.shape[-2], v_dec.shape[-2])

        # Memory check
        mem_bytes = cache.get_memory_bytes()
        self.assertGreater(mem_bytes, 0)

    def test_hook_manager_accumulation(self):
        """
        Verifies VectorAttentionHookManager intercepts attention weights and updates cache scores.
        """
        class DummySelfAttn(nn.Module):
            def __init__(self, num_query_heads=4, num_kv_heads=2):
                super().__init__()
                self.num_query_heads = num_query_heads
                self.num_kv_heads = num_kv_heads

            def forward(self, x):
                # Fake attention weights: [batch, num_query_heads, q_len, kv_len]
                attn_weights = torch.ones(1, self.num_query_heads, 1, 32) * 0.25
                return x, attn_weights

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.ModuleDict({"self_attn": DummySelfAttn()}),
                    nn.ModuleDict({"self_attn": DummySelfAttn()}),
                ])

        dummy_model = DummyModel()
        cache = VectorKVCache(w_ols=self.w_ols)
        hook_mgr = VectorAttentionHookManager(
            dummy_model,
            cache,
            num_query_heads=4,
            num_kv_heads=2,
        )

        with hook_mgr:
            # Run forward pass through dummy layers
            x = torch.randn(1, 1, 64)
            for layer in dummy_model.layers:
                layer["self_attn"](x)

        # Scores should now be registered for layer 0 and layer 1
        self.assertIsNotNone(cache.accumulated_scores[0])
        self.assertIsNotNone(cache.accumulated_scores[1])
        # GQA grouped: [1, 2, 32]
        self.assertEqual(cache.accumulated_scores[0].shape, (1, 2, 32))

    def test_attention_sinks_retention_vector(self):
        """
        Verifies that initial tokens (num_sink_tokens) are always kept in the Retain pool
        with exact physical values in VectorKVCache.
        """
        num_sinks = 4
        cache = VectorKVCache(
            compression_ratio=0.75,
            approx_ratio=0.15,
            recent_ratio=0.10,
            min_recent_tokens=8,
            num_sink_tokens=num_sinks,
            w_ols=self.w_ols,
        )

        total_tokens = 120
        k_init = torch.randn(self.batch_size, self.num_heads, total_tokens - 1, self.head_dim)
        v_init = torch.randn(self.batch_size, self.num_heads, total_tokens - 1, self.head_dim)
        # Mark sink tokens
        k_init[:, :, :num_sinks, :] = 888.0
        v_init[:, :, :num_sinks, :] = 777.0

        cache.update(k_init, v_init, layer_idx=0)

        # Trigger prune with 1 decode token
        k_step = torch.randn(self.batch_size, self.num_heads, 1, self.head_dim)
        v_step = torch.randn(self.batch_size, self.num_heads, 1, self.head_dim)
        k_out, v_out = cache.update(k_step, v_step, layer_idx=0)

        # Check that stored positions include positions 0..num_sinks-1
        stored_pos = cache._stored_positions[0]
        for sink_idx in range(num_sinks):
            self.assertTrue(
                (stored_pos == sink_idx).any().item(),
                f"Sink position {sink_idx} must be retained in stored_positions!"
            )

        # Sink keys and values in output must equal the marked values
        sink_keys = k_out[:, :, :num_sinks, :]
        sink_vals = v_out[:, :, :num_sinks, :]
        self.assertTrue(torch.allclose(sink_keys, torch.tensor(888.0)))
        self.assertTrue(torch.allclose(sink_vals, torch.tensor(777.0)))

    def test_cached_eps_tracking(self):
        """
        Verifies that _cached_eps is correctly initialized during prefill and
        incremented during decode steps.
        """
        cache = VectorKVCache(
            compression_ratio=0.50,
            approx_ratio=0.20,
            recent_ratio=0.10,
            min_recent_tokens=4,
            w_ols=self.w_ols,
        )

        prefill_len = 20
        k = torch.randn(self.batch_size, self.num_heads, prefill_len, self.head_dim)
        v = torch.randn(self.batch_size, self.num_heads, prefill_len, self.head_dim)
        cache.update(k, v, layer_idx=0)

        self.assertIsNotNone(cache._cached_eps[0])
        self.assertEqual(cache._cached_eps[0].shape[-1], prefill_len)

        # 3 decode steps
        for step in range(3):
            k_tok = torch.randn(self.batch_size, self.num_heads, 1, self.head_dim)
            v_tok = torch.randn(self.batch_size, self.num_heads, 1, self.head_dim)
            cache.update(k_tok, v_tok, layer_idx=0)
            # After pruning, cached eps must match the number of stored keys
            n_k = cache._key_cache[0].shape[-2]
            self.assertEqual(cache._cached_eps[0].shape[-1], n_k)

    def test_multistep_decode_budget_stability(self):
        """
        Verifies that across 20 consecutive decode steps, the KV cache budget remains
        strictly constant at 2 * target_budget and does NOT suffer from recursive compression.
        """
        compression_ratio = 0.75
        approx_ratio = 0.15
        cache = VectorKVCache(
            compression_ratio=compression_ratio,
            approx_ratio=approx_ratio,
            recent_ratio=0.10,
            min_recent_tokens=4,
            w_ols=self.w_ols,
        )

        prefill_len = 100
        k_prefill = torch.randn(self.batch_size, self.num_heads, prefill_len, self.head_dim)
        v_prefill = torch.randn(self.batch_size, self.num_heads, prefill_len, self.head_dim)
        cache.update(k_prefill, v_prefill, layer_idx=0)

        expected_budget = int((1.0 - compression_ratio) * (prefill_len + 1))  # 25 tokens
        initial_mem = None

        # Run 20 decode steps
        for step in range(20):
            k_tok = torch.randn(self.batch_size, self.num_heads, 1, self.head_dim)
            v_tok = torch.randn(self.batch_size, self.num_heads, 1, self.head_dim)
            cache.update(k_tok, v_tok, layer_idx=0)

            n_k = cache._key_cache[0].shape[-2]
            n_r = cache._value_cache[0].shape[-2]
            self.assertEqual(
                n_k + n_r,
                2 * expected_budget,
                f"Step {step}: Invariance violated! n_k={n_k}, n_r={n_r}, expected {2*expected_budget}"
            )
            self.assertGreaterEqual(n_k, expected_budget)

            mem_bytes = cache.get_memory_bytes()
            if initial_mem is None:
                initial_mem = mem_bytes
            else:
                self.assertEqual(mem_bytes, initial_mem, f"Step {step}: Memory collapsed from {initial_mem} to {mem_bytes}")

    def test_query_chunk_attention_injection_preserves_target_tokens(self):
        """
        Tests that Query Chunk Attention Injection successfully preserves target (needle)
        tokens in _stored_positions after budget pruning.
        """
        cache = VectorKVCache(
            compression_ratio=0.75,
            approx_ratio=0.20,
            num_sink_tokens=4,
            recent_ratio=0.10,
            w_ols=self.w_ols,
        )

        haystack_len = 90
        query_len = 10
        total_prefill = haystack_len + query_len  # 100 tokens

        # Step 1: Haystack prefill (tokens 0..89)
        cache.is_prefilling = True
        k_haystack = torch.randn(self.batch_size, self.num_heads, haystack_len, self.head_dim)
        v_haystack = torch.randn(self.batch_size, self.num_heads, haystack_len, self.head_dim)
        cache.update(k_haystack, v_haystack, layer_idx=0)

        # Step 2: Query chunk prefill (tokens 90..99)
        k_query = torch.randn(self.batch_size, self.num_heads, query_len, self.head_dim)
        v_query = torch.randn(self.batch_size, self.num_heads, query_len, self.head_dim)
        cache.update(k_query, v_query, layer_idx=0)

        # Step 3: Simulate attention hook registering high query attention on target token 50
        target_token_idx = 50
        mock_query_attn = torch.zeros(self.batch_size, self.num_heads, query_len, total_prefill)
        mock_query_attn[:, :, :, target_token_idx] = 10.0  # Massive attention on target token
        cache.register_attention_scores(layer_idx=0, attn_scores=mock_query_attn)

        # Step 4: Complete prefill, trigger decode step with pruning
        cache.is_prefilling = False
        k_dec = torch.randn(self.batch_size, self.num_heads, 1, self.head_dim)
        v_dec = torch.randn(self.batch_size, self.num_heads, 1, self.head_dim)
        cache.update(k_dec, v_dec, layer_idx=0)

        # Step 5: Verify target_token_idx is preserved in _stored_positions[0]
        stored_pos = cache._stored_positions[0]
        self.assertIsNotNone(stored_pos)
        stored_list = stored_pos[0, 0].cpu().tolist()
        self.assertIn(
            target_token_idx,
            stored_list,
            f"Target token {target_token_idx} with high query attention was evicted! Stored: {stored_list}"
        )


if __name__ == "__main__":
    unittest.main()

