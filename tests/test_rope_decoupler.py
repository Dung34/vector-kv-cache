import unittest
import torch
from src.calibration.rope_decoupler import rotate_half, apply_rope, decouple_rope


class TestRoPEDecoupler(unittest.TestCase):
    def test_rotate_half_property(self):
        # rotate_half twice is equivalent to negation: rotate_half(rotate_half(x)) == -x
        x = torch.randn(2, 4, 16, 128)
        x_rot2 = rotate_half(rotate_half(x))
        self.assertTrue(torch.allclose(x_rot2, -x, atol=1e-6))

    def test_rope_roundtrip_reconstruction(self):
        # Applying RoPE and then decouple_rope should reconstruct the exact original tensor
        batch_size = 2
        num_heads = 4
        seq_len = 128
        head_dim = 128

        k_orig = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)

        # Generate standard RoPE frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, :]  # [1, seq_len, head_dim]
        sin = emb.sin()[None, :, :]  # [1, seq_len, head_dim]

        # 1. Forward RoPE
        k_post_rope = apply_rope(k_orig, cos, sin)

        # 2. Inverse RoPE (Decoupling)
        k_reconstructed = decouple_rope(k_post_rope, cos, sin)

        # Maximum absolute error should be virtually zero (< 1e-5)
        max_err = (k_reconstructed - k_orig).abs().max().item()
        self.assertLess(max_err, 1e-5, f"Reconstruction error too high: {max_err}")

    def test_float16_precision(self):
        # Test in half precision
        k_orig = torch.randn(1, 2, 64, 128, dtype=torch.float16)
        half_angles = torch.randn(1, 64, 64, dtype=torch.float32)
        angles = torch.cat((half_angles, half_angles), dim=-1)
        cos = angles.cos().to(torch.float16)
        sin = angles.sin().to(torch.float16)

        k_rot = apply_rope(k_orig, cos, sin)
        k_rec = decouple_rope(k_rot, cos, sin)

        max_err = (k_rec - k_orig).abs().max().item()
        self.assertLess(max_err, 5e-3)


if __name__ == "__main__":
    unittest.main()
