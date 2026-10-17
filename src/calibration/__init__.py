"""
VECTOR Calibration Module.
Provides RoPE Decoupling and Per-Head OLS Regression for KV Cache Value Reconstruction.
"""

from .rope_decoupler import rotate_half, apply_rope, decouple_rope
from .ols_solver import PerHeadOLSSolver
from .collector import ActivationCollector

__all__ = [
    "rotate_half",
    "apply_rope",
    "decouple_rope",
    "PerHeadOLSSolver",
    "ActivationCollector",
]
