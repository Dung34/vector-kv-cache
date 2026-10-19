"""
Vector Budget Partitioner Module.
Implements the 3-tier partitioning algorithm (Retain / Approximate / Evict)
and mathematical Budget Swap invariance (N_K + N_R == 2 * target_budget).
"""

from typing import Optional, Tuple, NamedTuple, Any
import torch


class PartitionResult(NamedTuple):
    k_cand: torch.Tensor
    v_retain: torch.Tensor
    pos_cand: torch.Tensor
    retain_rel: torch.Tensor
    approx_rel: torch.Tensor
    cand_indices: torch.Tensor


class VectorBudgetPartitioner:
    """
    Manages budget allocation and token classification into 3 pools:
      - Retain (R): Keeps real K and real V (sinks + recent + highest eps_i).
      - Approximate (A): Keeps real K, evicts real V (lowest eps_i).
      - Evict (E): Evicts both K and V.
    """

    def __init__(
        self,
        compression_ratio: float = 0.75,
        approx_ratio: float = 0.20,
        recent_ratio: float = 0.10,
        min_recent_tokens: int = 32,
        num_sink_tokens: int = 4,
    ):
        assert 0.0 <= compression_ratio < 1.0, "compression_ratio (p_c) must be in [0.0, 1.0)"
        assert 0.0 <= approx_ratio < (1.0 - compression_ratio), (
            f"approx_ratio (p_a={approx_ratio}) must be < (1 - p_c={1.0 - compression_ratio})"
        )

        self.compression_ratio = compression_ratio
        self.approx_ratio = approx_ratio
        self.recent_ratio = recent_ratio
        self.min_recent_tokens = min_recent_tokens
        self.num_sink_tokens = num_sink_tokens
        self._cached_sink_indices: Optional[torch.Tensor] = None
        self._cached_recent_offsets: Optional[torch.Tensor] = None

    def compute_budgets(self, total_len: int, target_budget: int) -> Tuple[int, int, int, int, int]:
        """
        Computes the token allocation across Sinks, Recent, Retain, Keys, and Approximate pools.
        Guarantees strict physical memory invariance: N_K + N_R == 2 * target_budget.
        """
        # Sink tokens (always kept in Retain)
        k_sink = min(self.num_sink_tokens, max(0, target_budget // 4))

        # Recent tokens (always kept in Retain)
        k_recent = max(
            int(target_budget * self.recent_ratio),
            min(self.min_recent_tokens, (target_budget - k_sink) // 2),
        )
        k_recent = min(k_recent, total_len - k_sink)
        if k_sink + k_recent > target_budget:
            k_recent = max(1, target_budget - k_sink)

        # Retain value budget: N_R
        n_r = max(k_sink + k_recent, target_budget - int(self.approx_ratio * total_len))

        # Total keys budget: N_K = 2 * target_budget - N_R
        n_k = min(total_len, 2 * target_budget - n_r)
        if n_k + n_r > 2 * target_budget:
            n_r = max(k_sink + k_recent, 2 * target_budget - n_k)

        n_a = n_k - n_r
        return k_sink, k_recent, n_r, n_k, n_a

    def partition(
        self,
        layer_idx: int,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        pos_full: torch.Tensor,
        target_budget: int,
        accumulated_scores: Optional[torch.Tensor] = None,
        cached_eps: Optional[torch.Tensor] = None,
        reconstructor: Optional[Any] = None,
    ) -> PartitionResult:
        """
        Partitions candidate tokens and gathers tensors according to 3-tier policy.
        """
        batch_size, num_heads, total_len, head_dim = k_full.shape
        device = k_full.device

        k_sink, k_recent, n_r, n_k, n_a = self.compute_budgets(total_len, target_budget)

        cand_start = k_sink
        cand_end = total_len - k_recent
        cand_past_len = max(0, cand_end - cand_start)
        k_h2_needed = n_k - (k_sink + k_recent)

        # 1. Candidate pool selection via Accumulated Attention Scores
        if accumulated_scores is None or accumulated_scores.shape[-1] < cand_end:
            h2_indices = torch.arange(
                max(cand_start, cand_end - k_h2_needed),
                cand_end,
                device=device,
                dtype=torch.long,
            ).view(1, 1, -1).expand(batch_size, num_heads, -1)
        else:
            past_scores = accumulated_scores[..., cand_start:cand_end]
            _, h2_sub_idx = torch.topk(
                past_scores, k=min(k_h2_needed, cand_past_len), dim=-1, largest=True, sorted=False
            )
            h2_indices = cand_start + h2_sub_idx

        # 2. Fast error lookup via cached eps (O(1) lookups)
        n_r_h2 = n_r - (k_sink + k_recent)

        if n_r_h2 > 0 and n_a > 0 and h2_indices.shape[-1] > 0:
            if cached_eps is not None:
                eps_h2 = torch.gather(cached_eps, -1, h2_indices)
            elif reconstructor is not None:
                gather_h2_idx = h2_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                k_h2_cand = torch.gather(k_full, -2, gather_h2_idx)
                v_h2_cand = torch.gather(v_full, -2, gather_h2_idx)
                pos_h2_cand = torch.gather(pos_full, -1, h2_indices)
                eps_h2 = reconstructor.compute_reconstruction_error(layer_idx, k_h2_cand, v_h2_cand, pos_h2_cand)
            else:
                eps_h2 = torch.zeros_like(h2_indices, dtype=torch.float32)

            _, sorted_eps_idx = torch.sort(eps_h2, dim=-1, descending=True)
            retain_h2_subidx = sorted_eps_idx[..., :n_r_h2]
            approx_h2_subidx = sorted_eps_idx[..., n_r_h2:]

            retain_past_indices = torch.gather(h2_indices, -1, retain_h2_subidx)
            approx_past_indices = torch.gather(h2_indices, -1, approx_h2_subidx)
        elif n_r_h2 > 0:
            retain_past_indices = h2_indices
            approx_past_indices = torch.empty((batch_size, num_heads, 0), device=device, dtype=torch.long)
        else:
            retain_past_indices = torch.empty((batch_size, num_heads, 0), device=device, dtype=torch.long)
            approx_past_indices = h2_indices

        # 3. Concatenate Sinks, Retain H2, and Recent indices (reusing cached buffers)
        if (
            self._cached_sink_indices is None
            or self._cached_sink_indices.shape[-1] != k_sink
            or self._cached_sink_indices.device != device
        ):
            self._cached_sink_indices = torch.arange(
                0, k_sink, device=device, dtype=torch.long
            ).view(1, 1, k_sink)
        sink_indices = self._cached_sink_indices.expand(batch_size, num_heads, -1)

        if (
            self._cached_recent_offsets is None
            or self._cached_recent_offsets.shape[-1] != k_recent
            or self._cached_recent_offsets.device != device
        ):
            self._cached_recent_offsets = torch.arange(
                0, k_recent, device=device, dtype=torch.long
            ).view(1, 1, k_recent)
        recent_indices = cand_end + self._cached_recent_offsets.expand(batch_size, num_heads, -1)

        full_retain_indices = torch.cat([sink_indices, retain_past_indices, recent_indices], dim=-1)
        full_retain_indices, _ = torch.sort(full_retain_indices, dim=-1)
        full_approx_indices, _ = torch.sort(approx_past_indices, dim=-1)

        # Total Candidate keys of size N_K
        cand_indices = torch.cat([full_retain_indices, full_approx_indices], dim=-1)
        cand_indices, _ = torch.sort(cand_indices, dim=-1)

        # 4. Vectorized relative indices mapping (no python loops, no torch.nonzero)
        cand_expanded = cand_indices.unsqueeze(-1)
        retain_expanded = full_retain_indices.unsqueeze(-2)
        retain_mask = (cand_expanded == retain_expanded).any(dim=-1)

        _, ret_rel_unsorted = torch.topk(retain_mask.to(torch.int8), k=n_r, dim=-1, largest=True, sorted=False)
        retain_rel, _ = torch.sort(ret_rel_unsorted, dim=-1)

        if n_a > 0:
            _, app_rel_unsorted = torch.topk((~retain_mask).to(torch.int8), k=n_a, dim=-1, largest=True, sorted=False)
            approx_rel, _ = torch.sort(app_rel_unsorted, dim=-1)
        else:
            approx_rel = torch.empty((batch_size, num_heads, 0), device=device, dtype=torch.long)

        # 5. Gather persistent state tensors
        gather_k_idx = cand_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_cand = torch.gather(k_full, -2, gather_k_idx)
        gather_v_retain = full_retain_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        v_retain = torch.gather(v_full, -2, gather_v_retain)
        pos_cand = torch.gather(pos_full, -1, cand_indices)

        return PartitionResult(
            k_cand=k_cand,
            v_retain=v_retain,
            pos_cand=pos_cand,
            retain_rel=retain_rel,
            approx_rel=approx_rel,
            cand_indices=cand_indices,
        )
