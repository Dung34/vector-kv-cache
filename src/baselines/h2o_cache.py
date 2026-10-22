"""
Baseline 2: H2O (Heavy Hitter Oracle) Binary Eviction KV Cache.
Reference: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models", NeurIPS 2023.
"""

from typing import Optional, Tuple, Any, Dict, List
import torch

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    class DynamicCache:
        def __init__(self):
            pass


class H2OKVCache(DynamicCache):
    """
    H2O Binary Eviction KV Cache.
    Maintains a fixed budget of KV tokens split into:
      1. Recent window (k_recent): preserves local contextual coherence.
      2. Heavy Hitters (k_H2): tokens with the highest accumulated attention scores.
    Tokens outside these pools are permanently evicted (both K and V removed).
    """

    def __init__(
        self,
        compression_ratio: float = 0.75,     # p_c in [0.0, 1.0)
        recent_ratio: float = 0.10,          # percentage of budget reserved for recent tokens
        min_recent_tokens: int = 32,         # minimum number of recent tokens to keep
        num_sink_tokens: int = 4,            # attention sink (initial tokens) to keep permanently
        max_budget: Optional[int] = None,    # optional hard cap on number of tokens
        target_budget: Optional[int] = None,
        max_physical_budget: Optional[int] = None,
    ):
        try:
            super().__init__()
        except TypeError:
            pass

        assert 0.0 <= compression_ratio < 1.0, "compression_ratio (p_c) must be in [0.0, 1.0)"
        self.compression_ratio = compression_ratio
        self.budget_ratio = 1.0 - compression_ratio
        self.recent_ratio = recent_ratio
        self.min_recent_tokens = min_recent_tokens
        self.num_sink_tokens = num_sink_tokens
        if target_budget is not None and max_budget is not None and target_budget != max_budget:
            raise ValueError("target_budget and legacy max_budget must match when both are provided")
        self.target_budget = target_budget if target_budget is not None else max_budget
        self.max_budget = self.target_budget
        self.max_physical_budget = max_physical_budget
        if self.max_physical_budget is not None and self.max_physical_budget <= 0:
            raise ValueError("max_physical_budget must be positive")
        if (
            self.target_budget is not None
            and self.max_physical_budget is not None
            and self.target_budget > self.max_physical_budget
        ):
            raise ValueError("target_budget cannot exceed max_physical_budget")
        self.active_budget: Optional[int] = self.target_budget
        self.is_prefilling: bool = False
        self._seen_tokens: int = 0

        # Storage per layer
        self._key_cache: List[torch.Tensor] = []
        self._value_cache: List[torch.Tensor] = []
        self._stored_positions: List[Optional[torch.Tensor]] = []
        # Accumulated attention scores: [batch_size, num_kv_heads, seq_len]
        self.accumulated_scores: List[torch.Tensor] = []
        # Cached index buffers to eliminate repetitive GPU memory allocation
        self._cached_sink_indices: Optional[torch.Tensor] = None
        self._cached_recent_offsets: Optional[torch.Tensor] = None

    @property
    def name(self) -> str:
        return f"H2O-Cache(p_c={int(self.compression_ratio*100)}%)"

    @property
    def key_cache(self) -> List[torch.Tensor]:
        return self._key_cache

    @property
    def value_cache(self) -> List[torch.Tensor]:
        return self._value_cache

    def _physical_len(self, layer_idx: Optional[int] = 0) -> int:
        idx = 0 if layer_idx is None else layer_idx
        if not self._key_cache or idx >= len(self._key_cache) or self._key_cache[idx] is None:
            return 0
        return self._key_cache[idx].shape[-2]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if self._seen_tokens > 0:
            return self._seen_tokens
        return self._physical_len(layer_idx)

    def get_mask_sizes(
        self,
        cache_position: Any = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Return (kv_length, kv_offset) for causal attention mask generation.
        Uses _physical_len to ensure attention_mask matches physical key tensor length.
        """
        actual_layer_idx = 0
        query_len = 0

        if layer_idx is None and isinstance(cache_position, int):
            actual_layer_idx = cache_position
            query_len = 0
        else:
            if layer_idx is not None:
                actual_layer_idx = int(layer_idx)

            if cache_position is None:
                query_len = 0
            elif isinstance(cache_position, torch.Tensor):
                query_len = cache_position.shape[0] if cache_position.dim() > 0 else 1
            elif hasattr(cache_position, "shape") and len(cache_position.shape) > 0:
                query_len = cache_position.shape[0]
            elif hasattr(cache_position, "__len__"):
                query_len = len(cache_position)
            elif isinstance(cache_position, int):
                query_len = cache_position
            else:
                try:
                    query_len = int(cache_position)
                except Exception:
                    query_len = 0

        phys_len = self._physical_len(actual_layer_idx)
        if query_len == 0:
            kv_len = phys_len
        elif query_len > 1:
            kv_len = phys_len + query_len
        else:
            if self.active_budget is not None:
                kv_len = min(phys_len + 1, self.active_budget)
            else:
                kv_len = phys_len + 1
        return (kv_len, 0)

    def reset_scores(self):
        """Reset accumulated scores and storage for a new sequence generation."""
        self.accumulated_scores = []
        self._key_cache = []
        self._value_cache = []
        self._stored_positions = []
        self.active_budget = self.max_budget
        self.is_prefilling = False
        self._seen_tokens = 0
        self._cached_sink_indices = None
        self._cached_recent_offsets = None
        if hasattr(self, "layers"):
            self.layers = []

    def _apply_prefill_physical_cap(self, layer_idx: int):
        if self.max_physical_budget is None:
            return
        physical_len = self._physical_len(layer_idx)
        if physical_len <= self.max_physical_budget:
            return
        sink = min(self.num_sink_tokens, self.max_physical_budget)
        keep_recent = min(physical_len, self.max_physical_budget - sink)
        recent_start = physical_len - keep_recent
        indices = torch.cat([
            torch.arange(sink, device=self._key_cache[layer_idx].device),
            torch.arange(recent_start, physical_len, device=self._key_cache[layer_idx].device),
        ]).unique(sorted=True)
        self._key_cache[layer_idx] = self._key_cache[layer_idx].index_select(-2, indices)
        self._value_cache[layer_idx] = self._value_cache[layer_idx].index_select(-2, indices)
        self._stored_positions[layer_idx] = self._stored_positions[layer_idx].index_select(-1, indices)
        if self.accumulated_scores[layer_idx] is not None:
            self.accumulated_scores[layer_idx] = self.accumulated_scores[layer_idx].index_select(-1, indices)

    def register_attention_scores(self, layer_idx: int, attn_scores: torch.Tensor):
        """
        Called by forward hook to accumulate attention weights into scores.
        attn_scores: [batch_size, num_kv_heads, query_len, seq_len] or [batch_size, num_kv_heads, seq_len]
        """
        if attn_scores.dim() == 4:
            # Sum over query dimension (dim=-2)
            step_scores = attn_scores.sum(dim=-2)  # [batch_size, num_kv_heads, seq_len]
        else:
            step_scores = attn_scores

        if len(self.accumulated_scores) <= layer_idx:
            while len(self.accumulated_scores) < layer_idx:
                self.accumulated_scores.append(None)
            self.accumulated_scores.append(step_scores.clone())
        else:
            curr_scores = self.accumulated_scores[layer_idx]
            if curr_scores is None:
                self.accumulated_scores[layer_idx] = step_scores.clone()
                return

            seq_len = step_scores.shape[-1]
            if curr_scores.shape[-1] < seq_len:
                diff = seq_len - curr_scores.shape[-1]
                pad = torch.zeros(
                    (*curr_scores.shape[:-1], diff),
                    device=curr_scores.device,
                    dtype=curr_scores.dtype
                )
                curr_scores = torch.cat([curr_scores, pad], dim=-1)
            elif curr_scores.shape[-1] > seq_len:
                curr_scores = curr_scores[..., :seq_len]

            self.accumulated_scores[layer_idx] = curr_scores + step_scores

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new incoming key_states and value_states.
        Tensors have shape: [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, num_heads, incoming_len, head_dim = key_states.shape

        if layer_idx == 0:
            self._seen_tokens += incoming_len

        # Step 1: Initialize layer or concatenate incoming states
        if len(self._key_cache) <= layer_idx:
            while len(self._key_cache) < layer_idx:
                self._key_cache.append(None)
                self._value_cache.append(None)
            k_full = key_states
            v_full = value_states
            self._key_cache.append(k_full)
            self._value_cache.append(v_full)
            pos_init = torch.arange(incoming_len, device=key_states.device).view(1, 1, -1).expand(batch_size, num_heads, -1)
            self._stored_positions.append(pos_init)
            if len(self.accumulated_scores) <= layer_idx:
                while len(self.accumulated_scores) < layer_idx:
                    self.accumulated_scores.append(None)
                self.accumulated_scores.append(
                    torch.zeros((batch_size, num_heads, incoming_len), device=key_states.device, dtype=torch.float32)
                )
            self._apply_prefill_physical_cap(layer_idx)
            k_full = self._key_cache[layer_idx]
            v_full = self._value_cache[layer_idx]
        else:
            if self._key_cache[layer_idx] is None:
                k_full = key_states
                v_full = value_states
            else:
                k_full = torch.cat([self._key_cache[layer_idx], key_states], dim=-2)
                v_full = torch.cat([self._value_cache[layer_idx], value_states], dim=-2)

            self._key_cache[layer_idx] = k_full
            self._value_cache[layer_idx] = v_full

            prev_pos = self._stored_positions[layer_idx]
            if prev_pos is not None:
                pos_incoming = prev_pos[..., -1:] + torch.arange(1, incoming_len + 1, device=key_states.device).view(1, 1, -1)
                self._stored_positions[layer_idx] = torch.cat([prev_pos, pos_incoming], dim=-1)
            else:
                self._stored_positions[layer_idx] = torch.arange(k_full.shape[-2], device=key_states.device).view(1, 1, -1).expand(batch_size, num_heads, -1)

            # Pad accumulated_scores for new tokens if needed
            curr_scores = self.accumulated_scores[layer_idx]
            if curr_scores is not None:
                pad_len = k_full.shape[-2] - curr_scores.shape[-1]
                if pad_len > 0:
                    pad = torch.zeros(
                        (batch_size, num_heads, pad_len),
                        device=key_states.device,
                        dtype=curr_scores.dtype
                    )
                    self.accumulated_scores[layer_idx] = torch.cat([curr_scores, pad], dim=-1)

        total_len = k_full.shape[-2]

        # Step 2: Determine target token budget k
        if self.active_budget is None:
            if self.max_budget is not None:
                self.active_budget = min(self.max_budget, int(self.budget_ratio * total_len))
            else:
                self.active_budget = int(self.budget_ratio * total_len)
            self.active_budget = max(self.active_budget, min(self.min_recent_tokens, total_len))
            if self.max_physical_budget is not None:
                self.active_budget = min(self.active_budget, self.max_physical_budget)

        if self.compression_ratio == 0.0 and self.target_budget is None:
            self.active_budget = total_len
        target_budget = self.active_budget

        # During prefill (incoming_len > 1 or is_prefilling) or before exceeding budget: Keep everything in Retain
        if incoming_len > 1 or getattr(self, "is_prefilling", False) or total_len <= target_budget:
            self._key_cache[layer_idx] = k_full
            self._value_cache[layer_idx] = v_full
            self._apply_prefill_physical_cap(layer_idx)
            k_full = self._key_cache[layer_idx]
            v_full = self._value_cache[layer_idx]
            self._sync_dynamic_layers(layer_idx, k_full, v_full)
            return k_full, v_full

        # Step 3: Split budget into Sinks, Recent, and Heavy Hitters (H2)
        k_sink = min(self.num_sink_tokens, max(0, target_budget // 4))
        k_recent = max(int(target_budget * self.recent_ratio), min(self.min_recent_tokens, (target_budget - k_sink) // 2))
        k_recent = min(k_recent, total_len - k_sink)
        if k_sink + k_recent > target_budget:
            k_recent = max(1, target_budget - k_sink)
        k_h2 = target_budget - k_sink - k_recent

        # Step 4: Identify candidate middle tokens (between sink and recent)
        cand_start = k_sink
        cand_end = total_len - k_recent
        cand_len = max(0, cand_end - cand_start)

        if cand_len <= k_h2:
            self._key_cache[layer_idx] = k_full
            self._value_cache[layer_idx] = v_full
            self._sync_dynamic_layers(layer_idx, k_full, v_full)
            return k_full, v_full

        # Sinks indices (reused from cache buffer to avoid GPU memory allocations)
        if (
            self._cached_sink_indices is None
            or self._cached_sink_indices.shape[-1] != k_sink
            or self._cached_sink_indices.device != key_states.device
        ):
            self._cached_sink_indices = torch.arange(
                0, k_sink, device=key_states.device, dtype=torch.long
            ).view(1, 1, k_sink)
        sink_indices = self._cached_sink_indices.expand(batch_size, num_heads, -1)

        # Recent indices for each batch & head (reused offset buffer)
        if (
            self._cached_recent_offsets is None
            or self._cached_recent_offsets.shape[-1] != k_recent
            or self._cached_recent_offsets.device != key_states.device
        ):
            self._cached_recent_offsets = torch.arange(
                0, k_recent, device=key_states.device, dtype=torch.long
            ).view(1, 1, k_recent)
        recent_indices = cand_end + self._cached_recent_offsets.expand(batch_size, num_heads, -1)

        # Top-k Heavy Hitters selection across candidate middle tokens
        if k_h2 > 0:
            scores = self.accumulated_scores[layer_idx][..., cand_start:cand_end]
            _, h2_sub_indices = torch.topk(scores, k=k_h2, dim=-1, largest=True, sorted=False)
            h2_sub_sorted, _ = torch.sort(h2_sub_indices, dim=-1)
            h2_indices = cand_start + h2_sub_sorted
            # Since sink_indices < h2_indices < recent_indices, concatenation is strictly monotonic
            selected_indices = torch.cat([sink_indices, h2_indices, recent_indices], dim=-1)
        else:
            selected_indices = torch.cat([sink_indices, recent_indices], dim=-1)

        # Step 5: Gather K and V using selected indices
        gather_indices = selected_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_pruned = torch.gather(k_full, -2, gather_indices)
        v_pruned = torch.gather(v_full, -2, gather_indices)

        self._key_cache[layer_idx] = k_pruned
        self._value_cache[layer_idx] = v_pruned
        if len(self._stored_positions) > layer_idx and self._stored_positions[layer_idx] is not None:
            self._stored_positions[layer_idx] = torch.gather(self._stored_positions[layer_idx], -1, selected_indices)
        self.accumulated_scores[layer_idx] = torch.gather(
            self.accumulated_scores[layer_idx], -1, selected_indices
        )

        self._sync_dynamic_layers(layer_idx, k_pruned, v_pruned)
        return k_pruned, v_pruned

    def _sync_dynamic_layers(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Synchronize self.layers if transformers relies on DynamicLayer objects."""
        if hasattr(self, "layers"):
            while len(self.layers) <= layer_idx:
                if self.layer_class_to_replicate is not None:
                    self.layers.append(self.layer_class_to_replicate())
                else:
                    break
            if len(self.layers) > layer_idx:
                self.layers[layer_idx].keys = k
                self.layers[layer_idx].values = v

    def get_memory_bytes(self) -> int:
        """Calculate total bytes occupied by KV tensors in memory."""
        total_bytes = 0
        for k in self._key_cache:
            if k is not None:
                total_bytes += k.numel() * k.element_size()
        for v in self._value_cache:
            if v is not None:
                total_bytes += v.numel() * v.element_size()
        return total_bytes
