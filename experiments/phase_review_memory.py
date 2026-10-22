import argparse
import hashlib
from pathlib import Path
import sys
from datetime import datetime, timezone

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.h2o_cache import H2OKVCache
from src.vector_cache.vector_cache import VectorKVCache


def tensor_bytes(value):
    if value is None:
        return 0
    if isinstance(value, list):
        return sum(tensor_bytes(item) for item in value)
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    return 0


def cache_state(cache):
    persistent = cache.get_memory_bytes()
    auxiliary = sum(
        tensor_bytes(getattr(cache, name, None))
        for name in ("accumulated_scores", "_cached_eps", "_stored_positions")
    )
    lengths = [None if item is None else item.shape[-2] for item in cache._key_cache]
    return persistent, auxiliary, lengths


def run_case(name, cache, prompt_len, steps):
    k = torch.zeros(1, 2, prompt_len, 8)
    v = torch.zeros_like(k)
    cache.update(k, v, layer_idx=0)
    print(name, "after_prefill", cache_state(cache))
    for step in range(steps):
        token = torch.zeros(1, 2, 1, 8)
        cache.update(token, token, layer_idx=0)
        print(name, f"after_decode_{step + 1}", cache_state(cache))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", required=True)
    args = parser.parse_args()
    script_hash = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    print(
        f"nonce={args.nonce} timestamp={datetime.now(timezone.utc).isoformat()} "
        f"python={sys.version.split()[0]} torch={torch.__version__} "
        f"device=cpu seed=0 sha256={script_hash}"
    )
    torch.manual_seed(0)
    run_case(
        "h2o",
        H2OKVCache(compression_ratio=0.75, max_budget=16),
        prompt_len=64,
        steps=3,
    )
    run_case(
        "vector",
        VectorKVCache(
            compression_ratio=0.75,
            approx_ratio=0.15,
            min_recent_tokens=4,
            max_budget=16,
            w_ols=torch.eye(8).unsqueeze(0).unsqueeze(0),
        ),
        prompt_len=64,
        steps=3,
    )


if __name__ == "__main__":
    main()
