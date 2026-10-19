"""
Unit tests for VectorValueReconstructor.
Tests RoPE decoupling, prediction, error metric, and full V assembly.
"""

import unittest
import torch
from src.vector_cache.reconstructor import VectorValueReconstructor


class TestVectorValueReconstructor(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1
        self.num_layers = 2
        self.num_heads = 2
        self.head_dim = 32
        self.seq_len = 16

        torch.manual_seed(42)
        self.w_ols = torch.randn(self.num_layers, self.num_heads, self.head_dim, self.head_dim)
        self.reconstructor = VectorValueReconstructor(w_ols=self.w_ols)

    def test_predict_and_error_computation(self):
        k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        pos = torch.arange(self.seq_len).view(1, 1, -1).expand(self.batch_size, self.num_heads, -1)

        # Predict V
        v_hat = self.reconstructor.predict_v(0, k, pos)
        self.assertEqual(v_hat.shape, k.shape)

        # If v_real equals v_hat, error should be zero
        eps_zero = self.reconstructor.compute_reconstruction_error(0, k, v_hat, pos)
        self.assertTrue(torch.allclose(eps_zero, torch.zeros_like(eps_zero), atol=1e-5))

        # If v_real is noisy, error should be positive
        v_noisy = v_hat + 10.0
        eps_pos = self.reconstructor.compute_reconstruction_error(0, k, v_noisy, pos)
        self.assertTrue((eps_pos > 0).all())

    def test_reconstruct_full_v(self):
        k_stored = torch.randn(self.batch_size, self.num_heads, 4, self.head_dim)
        stored_pos = torch.arange(4).view(1, 1, -1).expand(self.batch_size, self.num_heads, -1)

        # 2 tokens retain (indices 0, 2), 2 tokens approx (indices 1, 3)
        retain_rel = torch.tensor([[[0, 2], [0, 2]]])
        approx_rel = torch.tensor([[[1, 3], [1, 3]]])
        v_retain = torch.full((self.batch_size, self.num_heads, 2, self.head_dim), 100.0)

        v_full = self.reconstructor.reconstruct_full_v(
            layer_idx=0,
            k_stored=k_stored,
            v_retain=v_retain,
            retain_rel=retain_rel,
            approx_rel=approx_rel,
            stored_pos=stored_pos,
            dtype=torch.float32,
        )

        self.assertEqual(v_full.shape, (self.batch_size, self.num_heads, 4, self.head_dim))
        # Retain positions must match v_retain exactly
        self.assertTrue(torch.allclose(v_full[:, :, 0, :], torch.tensor(100.0)))
        self.assertTrue(torch.allclose(v_full[:, :, 2, :], torch.tensor(100.0)))


if __name__ == "__main__":
    unittest.main()
