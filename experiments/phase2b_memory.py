import argparse
import datetime
import hashlib
import os
import random
import sys

import torch
import transformers

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.baselines.h2o_cache import H2OKVCache
from src.vector_cache.vector_cache import VectorKVCache


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(value):
    return value.numel() * value.element_size() if value is not None else 0


def auxiliary_bytes(cache):
    total = 0
    for name in ("accumulated_scores", "_stored_positions", "_cached_eps"):
        for value in getattr(cache, name, []):
            total += tensor_bytes(value)
    return total


def run_case(name, cache, states):
    cache.update(states, states, layer_idx=0)
    cache.update(states, states, layer_idx=0)
    physical = cache._physical_len(0)
    persistent = cache.get_memory_bytes()
    auxiliary = auxiliary_bytes(cache)
    cap = cache.max_physical_budget
    assert cap is None or physical <= cap, (name, physical, cap)
    print(
        f"case={name} shape={tuple(states.shape)} physical_len={physical} "
        f"persistent_bytes={persistent} auxiliary_bytes={auxiliary} "
        f"max_physical_budget={cap} expected_physical_le={cap} actual_physical={physical}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", required=True)
    args = parser.parse_args()
    script = os.path.abspath(__file__)
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    print(f"nonce={args.nonce}")
    print(
        f"timestamp={datetime.datetime.now().isoformat()} python={sys.version.split()[0]} "
        f"torch={torch.__version__} transformers={transformers.__version__} "
        f"device=cpu seed={seed} script_sha256={sha256_file(script)}"
    )
    states = torch.zeros(1, 2, 32, 4)
    run_case("h2o", H2OKVCache(target_budget=16, max_physical_budget=16), states)
    run_case(
        "vector",
        VectorKVCache(
            compression_ratio=0.5,
            approx_ratio=0.1,
            target_budget=16,
            max_physical_budget=32,
            w_ols=torch.eye(states.shape[-1]).unsqueeze(0).unsqueeze(0),
        ),
        states,
    )


if __name__ == "__main__":
    main()
