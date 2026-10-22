import argparse
import datetime
import hashlib
import os
import random
import sys

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.baselines.h2o_cache import H2OKVCache
from src.baselines.h2o_hook import H2OAttentionHookManager
from src.vector_cache.vector_cache import VectorKVCache
from src.vector_cache.vector_hook import VectorAttentionHookManager


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_variant(name, model, cache, manager_cls, prompt, steps):
    manager = manager_cls(model=model, kv_cache=cache)
    manager.register_hooks()
    with torch.no_grad():
        output = model(prompt, past_key_values=cache, use_cache=True)
        for step in range(1, steps + 1):
            token = output.logits[:, -1:].argmax(dim=-1)
            output = model(token, past_key_values=cache, use_cache=True)
            finite = bool(torch.isfinite(output.logits).all().item())
            physical = [cache._physical_len(layer_idx) for layer_idx in range(2)]
            bound = cache.max_physical_budget
            bounded = bound is not None and all(length <= bound for length in physical)
            print(
                f"variant={name} step={step} finite_logits={finite} "
                f"physical={physical} max_physical_budget={bound} bounded={bounded}"
            )
            assert finite, f"{name} produced non-finite logits at step {step}"
            assert bounded, f"{name} exceeded physical bound at step {step}"
    manager.remove_hooks()
    print(f"result={name} verdict=PASS steps={steps}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    config = AutoConfig.for_model(
        "qwen2",
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    prompt = torch.randint(0, config.vocab_size, (1, 64))
    w_ols = torch.eye(16).view(1, 1, 16, 16).expand(2, 2, 16, 16).clone()
    print(f"nonce={args.nonce}")
    print(
        f"timestamp={datetime.datetime.now().isoformat()} python={sys.version.split()[0]} "
        f"torch={torch.__version__} transformers={transformers.__version__} "
        f"device=cpu seed={seed} script_sha256={sha256_file(__file__)}"
    )
    for name, cache, manager_cls in [
        (
            "h2o",
            H2OKVCache(target_budget=32, max_physical_budget=64, min_recent_tokens=8, num_sink_tokens=4),
            H2OAttentionHookManager,
        ),
        (
            "vector",
            VectorKVCache(
                target_budget=32,
                max_physical_budget=64,
                compression_ratio=0.5,
                approx_ratio=0.1,
                min_recent_tokens=8,
                num_sink_tokens=4,
                w_ols=w_ols,
            ),
            VectorAttentionHookManager,
        ),
    ]:
        model = AutoModelForCausalLM.from_config(config).eval()
        run_variant(name, model, cache, manager_cls, prompt, args.steps)
    print("result=overall verdict=PASS")


if __name__ == "__main__":
    main()
