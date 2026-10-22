"""
VECTOR (Vector-based Asymmetric KV Cache Compression) Implementation.
Reference: Lin et al., "A Simple Plug-in for Improving Eviction-Based KV Cache Compression", 2024.

Refactored Architecture:
  1. VectorBudgetPartitioner: Handles 3-tier partitioning & Budget Swap invariance.
  2. VectorValueReconstructor: Handles RoPE decoupling & W_OLS value prediction.
  3. VectorKVCache: Coordinates Hugging Face DynamicCache lifecycle & inference runtime.
"""

from typing import Optional, Tuple, Any, Dict, List, Union
import torch

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    class DynamicCache:
        def __init__(self):
            pass

from .reconstructor import VectorValueReconstructor
from .partitioner import VectorBudgetPartitioner, PartitionResult


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
        num_sink_tokens: int = 4,            # attention sink (initial tokens) to keep permanently in Retain
        max_budget: Optional[int] = None,    # optional hard cap on number of tokens
        w_ols: Optional[Union[torch.Tensor, str, Dict[str, Any]]] = None,
        rope_theta: float = 1000000.0,       # default RoPE base frequency (Qwen2.5: 1M, Llama3: 500k)
        rotary_emb: Optional[Any] = None,
    ):
        try:
            super().__init__()
        except TypeError:
            pass

        self.compression_ratio = compression_ratio
        self.approx_ratio = approx_ratio
        self.recent_ratio = recent_ratio
        self.min_recent_tokens = min_recent_tokens
        self.num_sink_tokens = num_sink_tokens
        self.max_budget = max_budget
        self.active_budget = max_budget
        self.is_prefilling: bool = False
        self._seen_tokens: int = 0
        self.rope_theta = rope_theta
        self.rotary_emb = rotary_emb

        # Modular Subsystems
        self.partitioner = VectorBudgetPartitioner(
            compression_ratio=compression_ratio,
            approx_ratio=approx_ratio,
            recent_ratio=recent_ratio,
            min_recent_tokens=min_recent_tokens,
            num_sink_tokens=num_sink_tokens,
        )
        self.reconstructor = VectorValueReconstructor(
            w_ols=w_ols,
            rope_theta=rope_theta,
        )

        # Storage per layer
        self._key_cache: List[Optional[torch.Tensor]] = []
        self._value_cache: List[Optional[torch.Tensor]] = []
        self._retain_indices: List[Optional[torch.Tensor]] = []
        self._approx_indices: List[Optional[torch.Tensor]] = []
        self._stored_positions: List[Optional[torch.Tensor]] = []
        self._cached_eps: List[Optional[torch.Tensor]] = []
        self.accumulated_scores: List[Optional[torch.Tensor]] = []
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

    @property
    def w_ols(self) -> Optional[torch.Tensor]:
        return self.reconstructor.w_ols

    @w_ols.setter
    def w_ols(self, value: Optional[torch.Tensor]):
        self.reconstructor.w_ols = value

    def set_w_ols(self, w_ols: Union[torch.Tensor, str, Dict[str, Any]]):
        """Loads and formats the W_OLS weights tensor via Reconstructor."""
        self.reconstructor.set_w_ols(w_ols)

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
        Fully backward and forward compatible with all Hugging Face transformers versions:
          - get_mask_sizes(layer_idx)                [1 arg: int]
          - get_mask_sizes(cache_position)          [1 arg: Tensor]
          - get_mask_sizes(cache_position, layer_idx) [2 args: Tensor, int]
          - get_mask_sizes(query_len, layer_idx)     [2 args: int, int]
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
        elif self.active_budget is not None:
            kv_len = min(phys_len + 1, self.active_budget)
        else:
            kv_len = phys_len + 1
        return (kv_len, 0)

    def reset_scores(self):
        """Reset cache storage and accumulated scores for a new sequence generation."""
        self.accumulated_scores = []
        self._key_cache = []
        self._value_cache = []
        self._retain_indices = []
        self._approx_indices = []
        self._stored_positions = []
        self._cached_eps = []
        self.is_pruned = []
        self.active_budget = self.max_budget
        self.is_prefilling = False
        self._seen_tokens = 0
        if hasattr(self, "layers"):
            self.layers = []

    def register_attention_scores(self, layer_idx: int, attn_scores: torch.Tensor):
        """Called by forward hook to accumulate attention weights into scores."""
        if attn_scores.dim() == 4:
            step_scores = attn_scores.sum(dim=-2)
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

    def _ensure_layer_capacity(self, layer_idx: int):
        while len(self._key_cache) <= layer_idx:
            self._key_cache.append(None)
            self._value_cache.append(None)
            self._retain_indices.append(None)
            self._approx_indices.append(None)
            self._stored_positions.append(None)
            self._cached_eps.append(None)
            self.is_pruned.append(False)
            self.accumulated_scores.append(None)

    # --- Delegations to Reconstructor (Preserves Public & Protected API) ---
    def _compute_rope_cos_sin(self, positions: torch.Tensor, head_dim: int, device: torch.device, dtype: torch.dtype):
        return self.reconstructor.compute_rope_cos_sin(positions, head_dim, device, dtype)

    def _predict_v(self, layer_idx: int, k_rot: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.reconstructor.predict_v(layer_idx, k_rot, positions)

    def _compute_reconstruction_error(self, layer_idx: int, k_rot: torch.Tensor, v_real: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.reconstructor.compute_reconstruction_error(layer_idx, k_rot, v_real, positions)

    def _reconstruct_full_v(self, layer_idx: int, k_stored: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.reconstructor.reconstruct_full_v(
            layer_idx=layer_idx,
            k_stored=k_stored,
            v_retain=self._value_cache[layer_idx],
            retain_rel=self._retain_indices[layer_idx],
            approx_rel=self._approx_indices[layer_idx],
            stored_pos=self._stored_positions[layer_idx],
            dtype=dtype,
        )

    def _partition_and_prune(
        self,
        layer_idx: int,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        pos_full: torch.Tensor,
        target_budget: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Delegates partitioning to VectorBudgetPartitioner."""
        res: PartitionResult = self.partitioner.partition(
            layer_idx=layer_idx,
            k_full=k_full,
            v_full=v_full,
            pos_full=pos_full,
            target_budget=target_budget,
            accumulated_scores=self.accumulated_scores[layer_idx],
            cached_eps=self._cached_eps[layer_idx],
            reconstructor=self.reconstructor,
        )

        # Store persistent state
        self._key_cache[layer_idx] = res.k_cand
        self._value_cache[layer_idx] = res.v_retain
        self._stored_positions[layer_idx] = res.pos_cand
        self._retain_indices[layer_idx] = res.retain_rel
        self._approx_indices[layer_idx] = res.approx_rel
        self.is_pruned[layer_idx] = True

        # Prune cached eps and accumulated scores
        if self._cached_eps[layer_idx] is not None:
            self._cached_eps[layer_idx] = torch.gather(self._cached_eps[layer_idx], -1, res.cand_indices)
        if self.accumulated_scores[layer_idx] is not None:
            self.accumulated_scores[layer_idx] = torch.gather(self.accumulated_scores[layer_idx], -1, res.cand_indices)

        v_returned = self._reconstruct_full_v(layer_idx, self._key_cache[layer_idx], k_full.device, k_full.dtype)
        self._sync_dynamic_layers(layer_idx, self._key_cache[layer_idx], v_returned)
        return self._key_cache[layer_idx], v_returned

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
        if layer_idx == 0:
            self._seen_tokens += incoming_len
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
            self._cached_eps[layer_idx] = self._compute_reconstruction_error(layer_idx, key_states, value_states, pos_incoming)

            if self.accumulated_scores[layer_idx] is None:
                self.accumulated_scores[layer_idx] = torch.zeros((batch_size, num_heads, incoming_len), device=device, dtype=torch.float32)

            self._sync_dynamic_layers(layer_idx, key_states, value_states)
            return key_states, value_states

        # Compute error for incoming tokens once and cache it (O(1) during decode steps)
        incoming_eps = self._compute_reconstruction_error(layer_idx, key_states, value_states, pos_incoming)
        if self._cached_eps[layer_idx] is not None:
            self._cached_eps[layer_idx] = torch.cat([self._cached_eps[layer_idx], incoming_eps], dim=-1)
        else:
            self._cached_eps[layer_idx] = incoming_eps

        # Concatenate incoming states
        prev_k = self._key_cache[layer_idx]
        prev_pos = self._stored_positions[layer_idx]

        # Determine target budget (fixed once from prefill length, preventing recursive compression)
        total_len = prev_k.shape[-2] + incoming_len
        if self.active_budget is None:
            if self.max_budget is not None:
                self.active_budget = min(self.max_budget, int((1.0 - self.compression_ratio) * total_len))
            else:
                self.active_budget = int((1.0 - self.compression_ratio) * total_len)
            self.active_budget = max(self.active_budget, min(self.min_recent_tokens, total_len))

        target_budget = self.active_budget

        # During prefill (incoming_len > 1 or is_prefilling) or before exceeding budget: Keep everything in Retain
        if incoming_len > 1 or getattr(self, "is_prefilling", False) or total_len <= target_budget:
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
                pad = torch.zeros((batch_size, num_heads, pad_len), device=device, dtype=curr_scores.dtype)
                self.accumulated_scores[layer_idx] = torch.cat([curr_scores, pad], dim=-1)
        else:
            self.accumulated_scores[layer_idx] = torch.zeros(
                (batch_size, num_heads, k_full.shape[-2]), device=device, dtype=torch.float32
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
        """Calculates exact physical bytes occupied by persistent KV tensors in memory."""
        total_bytes = 0
        for k in self._key_cache:
            if k is not None:
                total_bytes += k.numel() * k.element_size()
        for v in self._value_cache:
            if v is not None:
                total_bytes += v.numel() * v.element_size()
        return total_bytes
