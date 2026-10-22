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
    return 0 if value is None else value.numel() * value.element_size()


def profile_case(name, cache, states):
    cache.update(states, states, layer_idx=0)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True,
    ) as profiler:
        with torch.profiler.record_function(f"{name}_update"):
            cache.update(states, states, layer_idx=0)

    cat_events = [event for event in profiler.key_averages() if "cat" in event.key]
    update_events = [event for event in profiler.key_averages() if event.key == f"{name}_update"]
    cat_self = sum(event.self_cpu_memory_usage for event in cat_events)
    cat_total = sum(event.cpu_memory_usage for event in cat_events)
    update_peak = max((event.cpu_memory_usage for event in update_events), default=0)
    persistent = cache.get_memory_bytes()
    print(
        f"case={name} input_shape={tuple(states.shape)} physical_len={cache._physical_len(0)} "
        f"cat_self_bytes={cat_self} cat_total_bytes={cat_total} "
        f"update_event_memory_bytes={update_peak} persistent_bytes={persistent}"
    )
    print(profiler.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", required=True)
    args = parser.parse_args()
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    script = os.path.abspath(__file__)
    print(f"nonce={args.nonce}")
    print(
        f"timestamp={datetime.datetime.now().isoformat()} python={sys.version.split()[0]} "
        f"torch={torch.__version__} transformers={transformers.__version__} "
        f"device=cpu seed={seed} script_sha256={sha256_file(script)}"
    )
    states = torch.zeros(1, 2, 32, 4)
    profile_case("h2o", H2OKVCache(target_budget=16, max_physical_budget=16), states)
    profile_case(
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
