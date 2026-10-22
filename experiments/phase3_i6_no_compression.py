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
from src.baselines.full_cache import FullKVCacheWrapper
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


def build_model(config):
    return AutoModelForCausalLM.from_config(config).eval()


def greedy_decode(model, cache, manager_cls, prompt, steps):
    manager = None
    if manager_cls is not None:
        manager = manager_cls(model=model, kv_cache=cache)
        manager.register_hooks()
    tokens = prompt.clone()
    with torch.no_grad():
        output = model(tokens, past_key_values=cache, use_cache=True)
        generated = []
        for _ in range(steps):
            token = output.logits[:, -1:].argmax(dim=-1)
            generated.append(token.item())
            output = model(token, past_key_values=cache, use_cache=True)
    if manager is not None:
        manager.remove_hooks()
    return generated


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
    base = build_model(config)
    prompt = torch.randint(0, config.vocab_size, (1, 16))
    models = [build_model(config), build_model(config), build_model(config)]
    for model in models:
        model.load_state_dict(base.state_dict())
    w_ols = torch.eye(16).view(1, 1, 16, 16).expand(2, 2, 16, 16).clone()
    variants = [
        ("full", FullKVCacheWrapper(), None),
        ("h2o_no_compression", H2OKVCache(compression_ratio=0.0), H2OAttentionHookManager),
        (
            "vector_no_compression",
            VectorKVCache(compression_ratio=0.0, approx_ratio=0.1, w_ols=w_ols),
            VectorAttentionHookManager,
        ),
    ]
    results = {}
    print(f"nonce={args.nonce}")
    print(
        f"timestamp={datetime.datetime.now().isoformat()} python={sys.version.split()[0]} "
        f"torch={torch.__version__} transformers={transformers.__version__} "
        f"device=cpu seed={seed} script_sha256={sha256_file(__file__)}"
    )
    for model, (name, cache, manager_cls) in zip(models, variants):
        results[name] = greedy_decode(model, cache, manager_cls, prompt, args.steps)
        print(f"variant={name} generated={results[name]}")
    for name in ("h2o_no_compression", "vector_no_compression"):
        equal = results[name] == results["full"]
        print(f"check=I6 variant={name} expected={results['full']} actual={results[name]} verdict={'PASS' if equal else 'FAIL'}")
        if not equal:
            raise AssertionError(f"I6 mismatch for {name}")
    print("result=PASS")


if __name__ == "__main__":
    main()
