"""
Per-Head Ordinary Least Squares (OLS) Solver with Ridge Regularization.
Reference: Lin et al., "A Simple Plug-in for Improving Eviction-Based KV Cache Compression", 2024.

Minimizes the K -> V prediction error in Value space:
    W_OLS = argmin_W || K * W^T - V ||_F^2 + lambda * || W ||_F^2

Solved via closed-form normal equations:
    W_OLS = (V^T * K) * (K^T * K + lambda * I)^(-1)

Supports online streaming covariance accumulation:
    Sigma_KK += K_batch^T * K_batch
    Sigma_VK += V_batch^T * K_batch
consuming only ~3.5 MB memory for all 28 layers.
"""

from typing import Optional, Dict, Tuple, List, Union
import torch


class PerHeadOLSSolver:
    """
    Fits independent OLS linear maps W_OLS^(l, h) for each layer l and KV head h.
    Shape of W_OLS per head: [head_dim, head_dim].
    Full weight tensor shape: [num_layers, num_kv_heads, head_dim, head_dim].
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        ridge_alpha: float = 1e-4,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.ridge_alpha = ridge_alpha
        self.device = torch.device(device)
        self.dtype = dtype

        # Covariance accumulators: [num_layers, num_kv_heads, head_dim, head_dim]
        self.sigma_kk = torch.zeros(
            (num_layers, num_kv_heads, head_dim, head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        self.sigma_vk = torch.zeros(
            (num_layers, num_kv_heads, head_dim, head_dim),
            device=self.device,
            dtype=self.dtype,
        )

        # Statistics for mean and variance: [num_layers, num_kv_heads, head_dim]
        self.sum_v = torch.zeros(
            (num_layers, num_kv_heads, head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        self.sum_v_sq = torch.zeros(
            (num_layers, num_kv_heads),
            device=self.device,
            dtype=self.dtype,
        )
        self.total_tokens: int = 0

        # Solved weights: [num_layers, num_kv_heads, head_dim, head_dim]
        self.w_ols: Optional[torch.Tensor] = None

    def reset(self):
        """Reset accumulators for a fresh calibration run."""
        self.sigma_kk.zero_()
        self.sigma_vk.zero_()
        self.sum_v.zero_()
        self.sum_v_sq.zero_()
        self.total_tokens = 0
        self.w_ols = None

    def update_batch(
        self,
        layer_idx: int,
        k_decoupled: torch.Tensor,
        v: torch.Tensor,
    ):
        """
        Accumulates covariance statistics for a specific layer.
        Args:
            layer_idx: index of transformer layer [0, num_layers - 1]
            k_decoupled: Decoupled Key states [batch, num_kv_heads, seq_len, head_dim]
            v: Value states [batch, num_kv_heads, seq_len, head_dim]
        """
        assert 0 <= layer_idx < self.num_layers, f"Invalid layer_idx {layer_idx}"
        batch_size, num_heads, seq_len, head_dim = k_decoupled.shape
        assert num_heads == self.num_kv_heads and head_dim == self.head_dim

        # Reshape to [num_kv_heads, N, head_dim] where N = batch_size * seq_len
        # Permute: [batch, num_heads, seq_len, head_dim] -> [num_heads, batch * seq_len, head_dim]
        k_flat = k_decoupled.permute(1, 0, 2, 3).reshape(num_heads, -1, head_dim).to(device=self.device, dtype=self.dtype)
        v_flat = v.permute(1, 0, 2, 3).reshape(num_heads, -1, head_dim).to(device=self.device, dtype=self.dtype)

        # Accumulate per-head covariance:
        # K^T * K: [num_heads, head_dim, N] @ [num_heads, N, head_dim] -> [num_heads, head_dim, head_dim]
        kk = torch.bmm(k_flat.transpose(1, 2), k_flat)
        # V^T * K: [num_heads, head_dim, N] @ [num_heads, N, head_dim] -> [num_heads, head_dim, head_dim]
        vk = torch.bmm(v_flat.transpose(1, 2), k_flat)

        self.sigma_kk[layer_idx] += kk
        self.sigma_vk[layer_idx] += vk

        # Track token count on layer 0 (or any layer)
        if layer_idx == 0:
            self.total_tokens += batch_size * seq_len

        # Accumulate sums for R^2 baseline
        self.sum_v[layer_idx] += v_flat.sum(dim=1)
        self.sum_v_sq[layer_idx] += (v_flat ** 2).sum(dim=(1, 2))

    def solve(self) -> torch.Tensor:
        """
        Solves the regularized normal equations for all layers and heads:
            W_OLS = Sigma_VK * (Sigma_KK + lambda * I)^(-1)
        Returns:
            w_ols: [num_layers, num_kv_heads, head_dim, head_dim]
        """
        eye = torch.eye(self.head_dim, device=self.device, dtype=self.dtype).unsqueeze(0).unsqueeze(0)
        # Regularized Sigma_KK: [num_layers, num_kv_heads, head_dim, head_dim]
        sigma_kk_reg = self.sigma_kk + self.ridge_alpha * eye

        # Solve for each layer and head using torch.linalg.solve:
        # A * X = B where A = Sigma_KK_reg^T = Sigma_KK_reg, B = Sigma_VK^T
        # Then W_OLS = X^T = Sigma_VK * Sigma_KK_reg^(-1)
        # torch.linalg.solve solves A @ X = B
        # Let's solve: (Sigma_KK_reg) @ W_OLS^T = Sigma_VK^T
        b = self.sigma_vk.transpose(-1, -2)
        # Solve on CPU using robust LAPACK kernels to prevent CUDA driver/kernel image mismatch on Tesla T4
        w_t = torch.linalg.solve(sigma_kk_reg.cpu(), b.cpu()).to(self.device)
        self.w_ols = w_t.transpose(-1, -2).to(torch.float32).contiguous()
        return self.w_ols

    def predict_v(
        self,
        layer_idx: int,
        k_decoupled: torch.Tensor,
        w_ols: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reconstructs Value vectors from decoupled Key vectors:
            V_hat = K_decoupled * W_OLS^T
        Args:
            layer_idx: layer index
            k_decoupled: [batch, num_kv_heads, seq_len, head_dim]
            w_ols: optional weights tensor. If None, uses self.w_ols.
        Returns:
            v_hat: [batch, num_kv_heads, seq_len, head_dim]
        """
        if w_ols is None:
            assert self.w_ols is not None, "Solver has not been solved yet. Call solve() first."
            weights = self.w_ols[layer_idx]  # [num_kv_heads, head_dim, head_dim]
        else:
            weights = w_ols[layer_idx]

        weights = weights.to(device=k_decoupled.device, dtype=k_decoupled.dtype)
        # k_decoupled: [batch, num_kv_heads, seq_len, head_dim]
        # weights: [num_kv_heads, head_dim, head_dim]
        # We want for each head h: v_hat[:, h, :, :] = k_decoupled[:, h, :, :] @ weights[h].T
        # Using torch.einsum: 'b h s d, h o d -> b h s o'
        v_hat = torch.einsum("bhsd, hod -> bhso", k_decoupled, weights)
        return v_hat

    def compute_r2(
        self,
        layer_idx: int,
        k_decoupled: torch.Tensor,
        v: torch.Tensor,
        w_ols: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Evaluates R^2 coefficient of determination on a held-out evaluation batch:
            R^2 = 1 - SS_res / SS_tot
        Returns:
            dict with 'head_r2' list and 'mean_r2' float for the layer.
        """
        with torch.no_grad():
            v_hat = self.predict_v(layer_idx, k_decoupled, w_ols=w_ols)

            # Flatten across batch and seq_len: [num_kv_heads, N, head_dim]
            num_heads = k_decoupled.shape[1]
            v_flat = v.permute(1, 0, 2, 3).reshape(num_heads, -1, self.head_dim).to(torch.float64)
            v_hat_flat = v_hat.permute(1, 0, 2, 3).reshape(num_heads, -1, self.head_dim).to(torch.float64)

            # SS_res: Sum of squared residuals
            ss_res = ((v_flat - v_hat_flat) ** 2).sum(dim=(1, 2))  # [num_kv_heads]

            # SS_tot: Total sum of squares around the mean
            v_mean = v_flat.mean(dim=1, keepdim=True)
            ss_tot = ((v_flat - v_mean) ** 2).sum(dim=(1, 2))  # [num_kv_heads]

            head_r2 = (1.0 - (ss_res / torch.clamp(ss_tot, min=1e-8))).tolist()
            mean_r2 = float(sum(head_r2) / len(head_r2))

        return {
            "head_r2": [round(r, 4) for r in head_r2],
            "mean_r2": round(mean_r2, 4),
        }
