"""
RoPE Decoupler Module (Inverse Rotary Position Embedding).
Reference: Lin et al., "A Simple Plug-in for Improving Eviction-Based KV Cache Compression", 2024.

Decouples position-dependent rotation from cached Key vectors:
    K_post-RoPE = R_m * K_pre-RoPE
    K_decoupled = R_m^(-1) * K_post-RoPE = R_m^T * K_post-RoPE

Since RoPE rotation matrix R_m is orthogonal (R_m^T * R_m = I),
the inverse rotation is equivalent to rotating by negative angle (-theta_m):
    cos(-theta) = cos(theta)
    sin(-theta) = -sin(theta)
Therefore:
    K_decoupled = (K_post-RoPE * cos) - (rotate_half(K_post-RoPE) * sin)
"""

from typing import Tuple, Optional
import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """
    Forward RoPE rotation (standard Hugging Face Transformers convention).
    k: [batch, num_heads, seq_len, head_dim]
    cos, sin: [batch, seq_len, head_dim] or [1, seq_len, head_dim]
    """
    if cos.dim() < k.dim():
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    return (k * cos) + (rotate_half(k) * sin)


def decouple_rope(
    k_rot: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """
    Inverse RoPE rotation (RoPE Decoupling).
    Restores the position-independent Key representation from RoPE-encoded Keys.

    Args:
        k_rot: Post-RoPE Key tensor [batch, num_heads, seq_len, head_dim]
        cos: Cosine positional tensor [batch, seq_len, head_dim] or [1, seq_len, head_dim]
        sin: Sine positional tensor [batch, seq_len, head_dim] or [1, seq_len, head_dim]
        unsqueeze_dim: Head dimension to broadcast cos/sin (default: 1)

    Returns:
        k_decoupled: Pre-RoPE / Decoupled Key tensor of same shape as k_rot.
    """
    if cos.dim() < k_rot.dim():
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    # Inverse rotation: R_{-m} = cos(m*theta)*I - sin(m*theta)*J
    return (k_rot * cos) - (rotate_half(k_rot) * sin)
