"""
VECTOR (Vector-based Asymmetric KV Cache Compression) Implementation.
Reference: Lin et al., "A Simple Plug-in for Improving Eviction-Based KV Cache Compression", 2024.

Implements asymmetric KV cache compression:
  1. Budget Swap / Budget Relaxation: With the same memory budget as H2O,
     retains more Keys: N_K = (1 - p_c + p_a) * L.
  2. 3-Tier Partitioning:
     - Retain (R): Keeps real K and real V for tokens with high reconstruction error eps_i
       and recent window tokens (k_recent).
     - Approximate (A): Keeps real K, discards physical V, reconstructs V_hat = W_OLS * (R_m^-1 * K)
       on-the-fly during attention forward pass.
     - Evict (E): Discards both K and V.
  3. Strict physical memory invariance:
     Memory_VECTOR == Memory_H2O == 2 * (1 - p_c) * L * D elements.
"""

from typing import Optional, Tuple, Any, Dict, List, Union
import torch
import torch.nn as nn

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    class DynamicCache:
        def __init__(self):
            pass

from ..calibration.rope_decoupler import decouple_rope


class VectorKVCache(DynamicCache):
    """
    Asymmetric KV Cache with 3-tier partitioning (Retain / Approximate / Evict).
    Subclasses Hugging Face DynamicCache for seamless integration with model.generate().
    """

    def __init__(
        self,
        compression_ratio: float = 0.75,     # p_c in [0.0, 1.0)
        approx_ratio: float = 0.20,          # p_a in [0.0, 1.0 - p_c)
        recent_ratio: float = 0.10,          # percentage of budget reserved for recent tokens
        min_recent_tokens: int = 32,         # minimum number of recent tokens to keep
        max_budget: Optional[int] = None,    # optional hard cap on number of tokens
        w_ols: Optional[Union[torch.Tensor, str, Dict[str, Any]]] = None,
        rope_theta: float = 1000000.0,       # default RoPE base frequency (Qwen2.5: 1M, Llama3: 500k)
        rotary_emb: Optional[Any] = None,
    ):
        try:
            super().__init__()
        except TypeError:
            pass

        assert 0.0 <= compression_ratio < 1.0, "compression_ratio (p_c) must be in [0.0, 1.0)"
        assert 0.0 <= approx_ratio < (1.0 - compression_ratio), (
            f"approx_ratio (p_a={approx_ratio}) must be < (1 - p_c={1.0 - compression_ratio})"
        )

        self.compression_ratio = compression_ratio
        self.approx_ratio = approx_ratio
        self.recent_ratio = recent_ratio
        self.min_recent_tokens = min_recent_tokens
        self.max_budget = max_budget
        self.rope_theta = rope_theta
        self.rotary_emb = rotary_emb

        # Load W_OLS if provided
        self.w_ols: Optional[torch.Tensor] = None
        if w_ols is not None:
            self.set_w_ols(w_ols)

        # Storage per layer
        # _key_cache stores physical K tensors: shape [batch, num_heads, N_K, head_dim]
        self._key_cache: List[Optional[torch.Tensor]] = []
        # _value_cache stores physical V_retain tensors: shape [batch, num_heads, N_R, head_dim]
        self._value_cache: List[Optional[torch.Tensor]] = []
        # Partition indices for each layer:
        # retain_indices: relative indices in 0..N_K-1 that belong to Retain pool [batch, num_heads, N_R]
        self._retain_indices: List[Optional[torch.Tensor]] = []
        # approx_indices: relative indices in 0..N_K-1 that belong to Approximate pool [batch, num_heads, N_A]
        self._approx_indices: List[Optional[torch.Tensor]] = []
        # Original sequence positions of stored keys: [batch, num_heads, N_K]
        self._stored_positions: List[Optional[torch.Tensor]] = []

        # Accumulated attention scores: [batch, num_heads, seq_len]
        self.accumulated_scores: List[Optional[torch.Tensor]] = []

        # Tracking state
        self.is_pruned: List[bool] = []

    @property
    def name(self) -> str:
        return f"VECTOR-Cache(p_c={int(self.compression_ratio*100)}%, p_a={int(self.approx_ratio*100)}%)"

    @property
    def key_cache(self) -> List[torch.Tensor]:
        return self._key_cache

    @property
    def value_cache(self) -> List[torch.Tensor]:
        return self._value_cache

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

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if not self._key_cache or layer_idx >= len(self._key_cache) or self._key_cache[layer_idx] is None:
            return 0
        return self._key_cache[layer_idx].shape[-2]

    def reset_scores(self):
        """Reset cache storage and accumulated scores for a new sequence generation."""
        self.accumulated_scores = []
        self._key_cache = []
        self._value_cache = []
        self._retain_indices = []
        self._approx_indices = []
        self._stored_positions = []
        self.is_pruned = []
        if hasattr(self, "layers"):
            self.layers = []

    def register_attention_scores(self, layer_idx: int, attn_scores: torch.Tensor):
        """
        Called by forward hook to accumulate attention weights into scores.
        attn_scores: [batch_size, num_kv_heads, query_len, seq_len] or [batch_size, num_kv_heads, seq_len]
        """
        if attn_scores.dim() == 4:
            step_scores = attn_scores.sum(dim=-2)  # [batch_size, num_kv_heads, seq_len]
        else:
            step_scores = attn_scores

        while len(self.accumulated_scores) <= layer_idx:
            self.accumulated_scores.append(None)

        curr_scores = self.accumulated_scores[layer_idx]
        if curr_scores is None:
            self.accumulated_scores[layer_idx] = step_scores.clone()
            return

        seq_len = step_scores.shape[-1]
        if curr_scores.shape[-1] < seq_len:
            diff = seq_len - curr_scores.shape[-1]
            pad = torch.zeros((*curr_scores.shape[:-1], diff), device=curr_scores.device, dtype=curr_scores.dtype)
            curr_scores = torch.cat([curr_scores, pad], dim=-1)
        elif curr_scores.shape[-1] > seq_len:
            curr_scores = curr_scores[..., :seq_len]

        self.accumulated_scores[layer_idx] = curr_scores + step_scores

    def _compute_rope_cos_sin(
        self,
        positions: torch.Tensor,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates standard RoPE (cos, sin) tensors for given position indices.
        positions: [batch_size, seq_len] or [batch_size, num_heads, seq_len]
        Returns: cos, sin of shape matching positions with head_dim
        """
        if positions.dim() == 3:
            # Flatten to 2D for computation then reshape back
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

    def _predict_v(
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
        cos, sin = self._compute_rope_cos_sin(positions, head_dim, device, dtype)
        # Cos/Sin have shape [batch, num_heads, seq_len, head_dim]
        # Decouple RoPE
        k_decoupled = decouple_rope(k_rot, cos, sin, unsqueeze_dim=1)

        # 2. OLS mapping
        if self.w_ols is not None:
            w_layer = self.w_ols[layer_idx].to(device=device, dtype=dtype)
            # w_layer shape: [num_heads, head_dim, head_dim]
            # k_decoupled: [batch, num_heads, seq_len, head_dim]
            # V_hat = k_decoupled @ W^T
            v_hat = torch.einsum("bhsd, hod -> bhso", k_decoupled, w_layer)
        else:
            # Fallback if W_OLS is not loaded: return k_decoupled as heuristic
            v_hat = k_decoupled

        return v_hat

    def _compute_reconstruction_error(
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
        v_hat = self._predict_v(layer_idx, k_rot, positions)
        diff = v_real - v_hat
        diff_norm = torch.norm(diff, p=2, dim=-1)
        v_norm = torch.norm(v_real, p=2, dim=-1)
        eps = diff_norm / (v_norm + 1e-6)
        return eps

    def _ensure_layer_capacity(self, layer_idx: int):
        while len(self._key_cache) <= layer_idx:
            self._key_cache.append(None)
            self._value_cache.append(None)
            self._retain_indices.append(None)
            self._approx_indices.append(None)
            self._stored_positions.append(None)
            self.is_pruned.append(False)
            self.accumulated_scores.append(None)

    def _partition_and_prune(
        self,
        layer_idx: int,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        pos_full: torch.Tensor,
        target_budget: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes the 3-tier partitioning algorithm:
          1. Candidate pool selection (Budget Relaxation): top N_K tokens.
          2. Error calculation: eps_i = || V - W_OLS * K ||.
          3. Partitioning:
             - Retain (R): recent tokens + candidate tokens with highest eps_i (total N_R).
             - Approximate (A): candidate tokens with lowest eps_i (total N_A).
             - Evict (E): remaining tokens.
        """
        batch_size, num_heads, total_len, head_dim = k_full.shape
        device = k_full.device

        # Step 1: Compute budgets with mathematical invariance
        # Target H2O budget in token pairs is target_budget
        # VECTOR Key budget: N_K = min(total_len, target_budget + int(self.approx_ratio * total_len))
        # Recent tokens count:
        k_recent = max(int(target_budget * self.recent_ratio), min(self.min_recent_tokens, target_budget // 2))
        k_recent = min(k_recent, total_len)

        # Value retain budget: N_R
        # N_R = max(k_recent, target_budget - int(self.approx_ratio * total_len))
        n_r = max(k_recent, target_budget - int(self.approx_ratio * total_len))
        # Total keys budget: N_K = 2 * target_budget - N_R (guarantees N_K + N_R == 2 * target_budget)
        n_k = min(total_len, 2 * target_budget - n_r)
        # Readjust n_r if n_k hit total_len boundary
        if n_k + n_r > 2 * target_budget:
            n_r = max(k_recent, 2 * target_budget - n_k)

        n_a = n_k - n_r

        # Step 2: Candidate pool selection via Accumulated Attention Scores
        # Candidate past pool length (excluding recent window)
        cand_past_len = total_len - k_recent
        k_h2_needed = n_k - k_recent

        scores = self.accumulated_scores[layer_idx]
        if scores is None or scores.shape[-1] < cand_past_len:
            # If no scores available yet, default to most recent tokens
            h2_indices = torch.arange(
                max(0, cand_past_len - k_h2_needed),
                cand_past_len,
                device=device,
                dtype=torch.long
            ).view(1, 1, -1).expand(batch_size, num_heads, -1)
        else:
            past_scores = scores[..., :cand_past_len]
            _, h2_indices = torch.topk(past_scores, k=k_h2_needed, dim=-1, largest=True, sorted=False)

        # Recent indices: [batch, num_heads, k_recent]
        recent_indices = torch.arange(
            cand_past_len,
            total_len,
            device=device,
            dtype=torch.long
        ).view(1, 1, k_recent).expand(batch_size, num_heads, -1)

        # Candidate indices pool of size N_K: [batch, num_heads, N_K]
        cand_indices = torch.cat([h2_indices, recent_indices], dim=-1)
        cand_indices, _ = torch.sort(cand_indices, dim=-1)

        # Gather candidate tensors
        gather_k_idx = cand_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_cand = torch.gather(k_full, -2, gather_k_idx)
        v_cand = torch.gather(v_full, -2, gather_k_idx)
        pos_cand = torch.gather(pos_full, -1, cand_indices)

        # Step 3: Compute reconstruction error eps_i on past candidate tokens (h2_indices only)
        # Recent tokens are forced into Retain, so we partition the remaining (N_K - k_recent) tokens
        gather_h2_idx = h2_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_h2_cand = torch.gather(k_full, -2, gather_h2_idx)
        v_h2_cand = torch.gather(v_full, -2, gather_h2_idx)
        pos_h2_cand = torch.gather(pos_full, -1, h2_indices)

        eps_h2 = self._compute_reconstruction_error(layer_idx, k_h2_cand, v_h2_cand, pos_h2_cand)

        # Number of H2 tokens to put into Retain
        n_r_h2 = n_r - k_recent

        if n_r_h2 > 0 and n_a > 0:
            # Highest eps -> Retain (hardest to approximate)
            # Lowest eps -> Approximate (collinear, easy to approximate)
            _, sorted_eps_idx = torch.sort(eps_h2, dim=-1, descending=True)
            retain_h2_subidx = sorted_eps_idx[..., :n_r_h2]
            approx_h2_subidx = sorted_eps_idx[..., n_r_h2:]

            retain_past_indices = torch.gather(h2_indices, -1, retain_h2_subidx)
            approx_past_indices = torch.gather(h2_indices, -1, approx_h2_subidx)
        elif n_r_h2 > 0 and n_a == 0:
            retain_past_indices = h2_indices
            approx_past_indices = torch.empty((batch_size, num_heads, 0), device=device, dtype=torch.long)
        else:
            retain_past_indices = torch.empty((batch_size, num_heads, 0), device=device, dtype=torch.long)
            approx_past_indices = h2_indices

        # Full Retain tokens = retain_past_indices + recent_indices
        full_retain_global_indices = torch.cat([retain_past_indices, recent_indices], dim=-1)
        full_retain_global_indices, _ = torch.sort(full_retain_global_indices, dim=-1)

        full_approx_global_indices, _ = torch.sort(approx_past_indices, dim=-1)

        # Stored keys are the cand_indices of length N_K
        # Now find relative positions of retain and approx tokens inside cand_indices (0 .. N_K - 1)
        # Using searchsorted or equality check
        cand_expanded = cand_indices.unsqueeze(-1)  # [B, H, N_K, 1]
        retain_expanded = full_retain_global_indices.unsqueeze(-2)  # [B, H, 1, N_R]
        retain_mask = (cand_expanded == retain_expanded).any(dim=-1)  # [B, H, N_K]

        # Store persistent state
        self._key_cache[layer_idx] = k_cand
        gather_v_retain = full_retain_global_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        self._value_cache[layer_idx] = torch.gather(v_full, -2, gather_v_retain)
        self._stored_positions[layer_idx] = pos_cand
        self.is_pruned[layer_idx] = True

        # Map indices within k_cand
        # retain_rel_indices: shape [B, H, N_R]
        # approx_rel_indices: shape [B, H, N_A]
        retain_rel_list = []
        approx_rel_list = []
        for b in range(batch_size):
            b_ret_heads = []
            b_app_heads = []
            for h in range(num_heads):
                ret_idx = torch.nonzero(retain_mask[b, h], as_tuple=False).squeeze(-1)
                app_idx = torch.nonzero(~retain_mask[b, h], as_tuple=False).squeeze(-1)
                b_ret_heads.append(ret_idx)
                b_app_heads.append(app_idx)
            retain_rel_list.append(torch.stack(b_ret_heads, dim=0))
            approx_rel_list.append(torch.stack(b_app_heads, dim=0))

        self._retain_indices[layer_idx] = torch.stack(retain_rel_list, dim=0)
        self._approx_indices[layer_idx] = torch.stack(approx_rel_list, dim=0)

        # Update accumulated scores to match candidate tokens
        if scores is not None:
            self.accumulated_scores[layer_idx] = torch.gather(scores, -1, cand_indices)

        # Reconstruct full V of size N_K to return for this attention step
        v_returned = self._reconstruct_full_v(layer_idx, k_cand, device, k_full.dtype)
        self._sync_dynamic_layers(layer_idx, k_cand, v_returned)
        return k_cand, v_returned

    def _reconstruct_full_v(
        self,
        layer_idx: int,
        k_stored: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Dynamically reconstructs full V of shape [batch, num_heads, N_K, head_dim]
        by placing stored real values for Retain tokens and on-the-fly W_OLS predictions for Approximate tokens.
        """
        batch_size, num_heads, n_k, head_dim = k_stored.shape
        v_out = torch.empty((batch_size, num_heads, n_k, head_dim), device=device, dtype=dtype)

        v_retain = self._value_cache[layer_idx]
        retain_rel = self._retain_indices[layer_idx]
        approx_rel = self._approx_indices[layer_idx]
        pos_stored = self._stored_positions[layer_idx]

        # 1. Place Retain values
        if v_retain is not None and retain_rel is not None and retain_rel.shape[-1] > 0:
            scatter_ret_idx = retain_rel.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            v_out.scatter_(-2, scatter_ret_idx, v_retain)

        # 2. Reconstruct and place Approximate values
        if approx_rel is not None and approx_rel.shape[-1] > 0:
            gather_app_k = approx_rel.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_approx = torch.gather(k_stored, -2, gather_app_k)
            pos_approx = torch.gather(pos_stored, -1, approx_rel)

            v_approx_hat = self._predict_v(layer_idx, k_approx, pos_approx)
            scatter_app_idx = approx_rel.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            v_out.scatter_(-2, scatter_app_idx, v_approx_hat.to(dtype))

        return v_out

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new incoming key_states and value_states.
        Tensors have shape: [batch_size, num_heads, incoming_len, head_dim]
        """
        batch_size, num_heads, incoming_len, head_dim = key_states.shape
        device = key_states.device
        dtype = key_states.dtype
        self._ensure_layer_capacity(layer_idx)

        # Track positions
        if cache_kwargs is not None and "position_ids" in cache_kwargs:
            pos_incoming = cache_kwargs["position_ids"]
            if pos_incoming.dim() == 2:
                pos_incoming = pos_incoming.unsqueeze(1).expand(-1, num_heads, -1)
        else:
            prev_pos = self._stored_positions[layer_idx]
            if prev_pos is None or prev_pos.shape[-1] == 0:
                pos_incoming = torch.arange(incoming_len, device=device, dtype=torch.long)
                pos_incoming = pos_incoming.view(1, 1, -1).expand(batch_size, num_heads, -1)
            else:
                last_pos = prev_pos[..., -1:]
                pos_incoming = last_pos + torch.arange(1, incoming_len + 1, device=device, dtype=torch.long).view(1, 1, -1)

        # Case 1: Initial empty layer
        if self._key_cache[layer_idx] is None:
            self._key_cache[layer_idx] = key_states
            self._value_cache[layer_idx] = value_states
            self._stored_positions[layer_idx] = pos_incoming
            self._retain_indices[layer_idx] = torch.arange(incoming_len, device=device).view(1, 1, -1).expand(batch_size, num_heads, -1)
            self._approx_indices[layer_idx] = torch.empty((batch_size, num_heads, 0), device=device, dtype=torch.long)
            self.is_pruned[layer_idx] = False

            if self.accumulated_scores[layer_idx] is None:
                self.accumulated_scores[layer_idx] = torch.zeros((batch_size, num_heads, incoming_len), device=device, dtype=torch.float32)

            self._sync_dynamic_layers(layer_idx, key_states, value_states)
            return key_states, value_states

        # Concatenate incoming states
        prev_k = self._key_cache[layer_idx]
        prev_pos = self._stored_positions[layer_idx]

        # Determine target budget
        total_len = prev_k.shape[-2] + incoming_len
        target_budget = int((1.0 - self.compression_ratio) * total_len)
        if self.max_budget is not None:
            target_budget = min(self.max_budget, target_budget)
        target_budget = max(target_budget, min(self.min_recent_tokens, total_len))

        # During prefill (incoming_len > 1) or before exceeding budget:
        # Keep everything in Retain
        if incoming_len > 1 or total_len <= target_budget:
            # Reconstruct previous full V if it was compressed, then concatenate
            if self.is_pruned[layer_idx]:
                v_full_prev = self._reconstruct_full_v(layer_idx, prev_k, device, dtype)
            else:
                v_full_prev = self._value_cache[layer_idx]

            k_full = torch.cat([prev_k, key_states], dim=-2)
            v_full = torch.cat([v_full_prev, value_states], dim=-2)
            pos_full = torch.cat([prev_pos, pos_incoming], dim=-1)

            self._key_cache[layer_idx] = k_full
            self._value_cache[layer_idx] = v_full
            self._stored_positions[layer_idx] = pos_full
            self._retain_indices[layer_idx] = torch.arange(k_full.shape[-2], device=device).view(1, 1, -1).expand(batch_size, num_heads, -1)
            self._approx_indices[layer_idx] = torch.empty((batch_size, num_heads, 0), device=device, dtype=torch.long)
            self.is_pruned[layer_idx] = False

            self._sync_dynamic_layers(layer_idx, k_full, v_full)
            return k_full, v_full

        # Decode step exceeding budget: Execute 3-Tier Partitioning and Pruning
        if self.is_pruned[layer_idx]:
            v_full_prev = self._reconstruct_full_v(layer_idx, prev_k, device, dtype)
        else:
            v_full_prev = self._value_cache[layer_idx]

        k_full = torch.cat([prev_k, key_states], dim=-2)
        v_full = torch.cat([v_full_prev, value_states], dim=-2)
        pos_full = torch.cat([prev_pos, pos_incoming], dim=-1)

        # Pad accumulated_scores to match k_full length
        curr_scores = self.accumulated_scores[layer_idx]
        if curr_scores is not None:
            pad_len = k_full.shape[-2] - curr_scores.shape[-1]
            if pad_len > 0:
                pad = torch.zeros(
                    (batch_size, num_heads, pad_len),
                    device=device,
                    dtype=curr_scores.dtype,
                )
                self.accumulated_scores[layer_idx] = torch.cat([curr_scores, pad], dim=-1)
        else:
            self.accumulated_scores[layer_idx] = torch.zeros(
                (batch_size, num_heads, k_full.shape[-2]),
                device=device,
                dtype=torch.float32,
            )

        return self._partition_and_prune(layer_idx, k_full, v_full, pos_full, target_budget)

    def _sync_dynamic_layers(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Synchronize self.layers if transformers relies on DynamicLayer objects."""
        if hasattr(self, "layers"):
            while len(self.layers) <= layer_idx:
                if getattr(self, "layer_class_to_replicate", None) is not None:
                    self.layers.append(self.layer_class_to_replicate())
                else:
                    break
            if len(self.layers) > layer_idx:
                self.layers[layer_idx].keys = k
                self.layers[layer_idx].values = v

    def get_memory_bytes(self) -> int:
        """
        Calculates exact physical bytes occupied by persistent KV tensors in memory.
        Only counts stored K and physically stored V_retain (excluding temporary tensors).
        """
        total_bytes = 0
        for k in self._key_cache:
            if k is not None:
                total_bytes += k.numel() * k.element_size()
        for v in self._value_cache:
            if v is not None:
                total_bytes += v.numel() * v.element_size()
        return total_bytes
