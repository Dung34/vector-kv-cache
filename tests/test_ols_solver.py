import unittest
import torch
from src.calibration.ols_solver import PerHeadOLSSolver


class TestPerHeadOLSSolver(unittest.TestCase):
    def test_synthetic_exact_linear_fit(self):
        # Generate synthetic data with known linear relationship: V = K @ W_true^T + noise
        num_layers = 2
        num_kv_heads = 2
        head_dim = 64
        batch_size = 2
        seq_len = 100

        solver = PerHeadOLSSolver(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            ridge_alpha=1e-5,
        )

        # Ground truth transformation: [num_layers, num_kv_heads, head_dim, head_dim]
        torch.manual_seed(42)
        w_true = torch.randn(num_layers, num_kv_heads, head_dim, head_dim) * 0.1

        # Train batches
        for layer_idx in range(num_layers):
            for _ in range(5):
                k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
                # V = K @ W_true^T (using einsum)
                v = torch.einsum("bhsd, hod -> bhso", k, w_true[layer_idx])
                # Add small noise (SNR > 40 dB)
                v = v + torch.randn_like(v) * 1e-4
                solver.update_batch(layer_idx, k, v)

        # Solve
        w_fitted = solver.solve()
        self.assertEqual(w_fitted.shape, (num_layers, num_kv_heads, head_dim, head_dim))

        # Check weights recovery
        diff = (w_fitted - w_true).abs().max().item()
        self.assertLess(diff, 1e-2, f"Fitted weights deviate too much from ground truth: {diff}")

        # Check R^2 on a validation batch
        for layer_idx in range(num_layers):
            k_val = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
            v_val = torch.einsum("bhsd, hod -> bhso", k_val, w_true[layer_idx])
            metrics = solver.compute_r2(layer_idx, k_val, v_val)
            self.assertGreater(metrics["mean_r2"], 0.99, f"R^2 must be > 0.99, got {metrics['mean_r2']}")
            for hr in metrics["head_r2"]:
                self.assertGreater(hr, 0.99)

    def test_predict_v_shape(self):
        solver = PerHeadOLSSolver(num_layers=1, num_kv_heads=2, head_dim=32)
        k = torch.randn(1, 2, 10, 32)
        v = torch.randn(1, 2, 10, 32)
        solver.update_batch(0, k, v)
        solver.solve()

        v_hat = solver.predict_v(0, k)
        self.assertEqual(v_hat.shape, k.shape)


if __name__ == "__main__":
    unittest.main()
