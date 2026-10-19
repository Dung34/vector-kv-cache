"""
Vector Value Reconstructor Module.
Handles RoPE decoupling, linear projection using W_OLS, and error metric calculation.
"""

from typing import Optional, Tuple, Any, Dict, Union
import torch
from ..calibration.rope_decoupler import decouple_rope


class VectorValueReconstructor:
    """
    Handles on-the-fly reconstruction of Value vectors from Key vectors:
      1. RoPE Decoupling: K_decoupled = R_m^-1 * K_rot
      2. OLS Prediction: V_hat = K_decoupled @ W_OLS^T
      3. Reconstruction error metric: eps_i = || V - V_hat || / (|| V || + 1e-6)
      4. Sparse Value Assembly: merges physically retained V and reconstructed V_hat.
    """

    def __init__(
        self,
        w_ols: Optional[Union[torch.Tensor, str, Dict[str, Any]]] = None,
        rope_theta: float = 1000000.0,
    ):
        self.rope_theta = rope_theta
        self.w_ols: Optional[torch.Tensor] = None
        if w_ols is not None:
            self.set_w_ols(w_ols)

    def set_w_ols(self, w_ols: Union[torch.Tensor, str, Dict[str, Any]]):
        """Loads and formats the W_OLS weights tensor."""
        if isinstance(w_ols, str):
            loaded = torch.load(w_ols, map_location="cpu")
            if isinstance(loaded, dict) and "w_ols" in loaded:
                self.w_ols = loaded["w_ols"]
            elif isinstance(loaded, torch.Tensor):
                self.w_ols = loaded
            else:
                raise ValueError(f"Unrecognized format in weights file: {w_ols}")
        elif isinstance(w_ols, dict) and "w_ols" in w_ols:
            self.w_ols = w_ols["w_ols"]
        elif isinstance(w_ols, torch.Tensor):
            self.w_ols = w_ols
        else:
            raise TypeError(f"w_ols must be torch.Tensor, str path, or dict, got {type(w_ols)}")

    def compute_rope_cos_sin(
        self,
        positions: torch.Tensor,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates standard RoPE (cos, sin) tensors for given position indices.
        positions: [batch_size, seq_len] or [batch_size, num_heads, seq_len]
        Returns: cos, sin of shape [batch, num_heads, seq_len, head_dim]
        """
        if positions.dim() == 3:
            b, h, s = positions.shape
            pos_flat = positions.view(b * h, s)
        else:
            pos_flat = positions
            b, h, s = positions.shape[0], 1, positions.shape[1]

        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
        freqs = torch.einsum("bs, d -> bsd", pos_flat.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype).view(b, h, s, head_dim)
        sin = emb.sin().to(dtype).view(b, h, s, head_dim)
        return cos, sin

    def predict_v(
        self,
        layer_idx: int,
        k_rot: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstructs Value vectors from RoPE-rotated Key vectors:
          1. Decouples RoPE: K_decoupled = R_m^-1 * K_rot
          2. Applies W_OLS linear mapping: V_hat = K_decoupled * W_OLS^T
        """
        batch_size, num_heads, seq_len, head_dim = k_rot.shape
        device = k_rot.device
        dtype = k_rot.dtype

        # 1. RoPE Decoupling
        cos, sin = self.compute_rope_cos_sin(positions, head_dim, device, dtype)
        k_decoupled = decouple_rope(k_rot, cos, sin, unsqueeze_dim=1)

        # 2. OLS mapping
        if self.w_ols is not None:
            w_layer = self.w_ols[layer_idx].to(device=device, dtype=dtype)
            v_hat = torch.einsum("bhsd, hod -> bhso", k_decoupled, w_layer)
        else:
            v_hat = k_decoupled

        return v_hat

    def compute_reconstruction_error(
        self,
        layer_idx: int,
        k_rot: torch.Tensor,
        v_real: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the relative L2 reconstruction error eps_i for candidate tokens:
          eps_i = || v_real_i - v_hat_i ||_2 / (|| v_real_i ||_2 + 1e-6)
        Returns: eps of shape [batch_size, num_heads, candidate_len]
        """
        v_hat = self.predict_v(layer_idx, k_rot, positions)
        diff = v_real - v_hat
        diff_norm = torch.norm(diff, p=2, dim=-1)
        v_norm = torch.norm(v_real, p=2, dim=-1)
        eps = diff_norm / (v_norm + 1e-6)
        return eps

    def reconstruct_full_v(
        self,
        layer_idx: int,
        k_stored: torch.Tensor,
        v_retain: Optional[torch.Tensor],
        retain_rel: Optional[torch.Tensor],
        approx_rel: Optional[torch.Tensor],
        stored_pos: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Dynamically reconstructs full V of shape [batch, num_heads, N_K, head_dim]
        by placing stored real values for Retain tokens and on-the-fly W_OLS predictions for Approximate tokens.
        """
        batch_size, num_heads, n_k, head_dim = k_stored.shape
        device = k_stored.device
        v_out = torch.empty((batch_size, num_heads, n_k, head_dim), device=device, dtype=dtype)

        # 1. Place Retain values
        if v_retain is not None and retain_rel is not None and retain_rel.shape[-1] > 0:
            scatter_ret_idx = retain_rel.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            v_out.scatter_(-2, scatter_ret_idx, v_retain)

        # 2. Reconstruct and place Approximate values
        if approx_rel is not None and approx_rel.shape[-1] > 0 and stored_pos is not None:
            gather_app_k = approx_rel.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_approx = torch.gather(k_stored, -2, gather_app_k)
            pos_approx = torch.gather(stored_pos, -1, approx_rel)

            v_approx_hat = self.predict_v(layer_idx, k_approx, pos_approx)
            scatter_app_idx = approx_rel.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            v_out.scatter_(-2, scatter_app_idx, v_approx_hat.to(dtype))

        return v_out
