import argparse
import datetime
import hashlib
import os
import sys
from types import MethodType

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoConfig

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


def run_variant(name, cache, hook_cls, model, prompt_len, steps):
    records = {}
    histories = {}
    original_update = cache.update

    def tracked_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        histories.setdefault(layer_idx, []).append(key_states.detach().clone())
        return original_update(key_states, value_states, layer_idx, cache_kwargs)

    cache.update = MethodType(tracked_update, cache)

    def make_hook(layer_idx):
        def hook(module, args, kwargs):
            mask = kwargs.get("attention_mask")
            pos = kwargs.get("position_ids")
            records[layer_idx] = {
                "mask_len": mask.shape[-1] if mask is not None else None,
                "position_id": pos[0, -1].item() if pos is not None else None,
            }
        return hook

    hooks = [model.model.layers[i].register_forward_pre_hook(make_hook(i), with_kwargs=True) for i in range(2)]
    manager = hook_cls(model=model, kv_cache=cache)
    manager.register_hooks()
    # Keep attention-score hooks, but retain the real causal mask so I2 can be observed.
    if manager.hooks:
        manager.hooks[0].remove()
        manager.hooks.pop(0)
    prompt = torch.randint(0, model.config.vocab_size, (1, prompt_len))
    with torch.no_grad():
        model(prompt, past_key_values=cache, use_cache=True, output_attentions=True)

    for step in range(1, steps + 1):
        records.clear()
        past_seen = cache.get_seq_length(0)
        token = torch.randint(0, model.config.vocab_size, (1, 1))
        with torch.no_grad():
            model(token, past_key_values=cache, use_cache=True, output_attentions=True)

        physical_lengths = [cache._physical_len(i) for i in range(2)]
        scores_lengths = [
            cache.accumulated_scores[i].shape[-1]
            if i < len(cache.accumulated_scores) and cache.accumulated_scores[i] is not None
            else None
            for i in range(2)
        ]
        position_ids = [records[i]["position_id"] for i in range(2)]
        mask_lengths = [records[i]["mask_len"] for i in range(2)]
        stored_positions = [cache._stored_positions[i] for i in range(2)]

        i1 = all(value == prompt_len + step - 1 for value in position_ids)
        i2 = all(mask_lengths[i] == physical_lengths[i] for i in range(2))
        i3 = len(set(physical_lengths)) == 1
        i4 = all(scores_lengths[i] == physical_lengths[i] for i in range(2))
        i5 = True
        for layer_idx in range(2):
            all_k = torch.cat(histories[layer_idx], dim=-2)
            positions = stored_positions[layer_idx]
            expected_k = torch.gather(
                all_k,
                -2,
                positions.unsqueeze(-1).expand(*positions.shape, all_k.shape[-1]),
            )
            actual_k = cache._key_cache[layer_idx]
            i5 = i5 and torch.equal(actual_k, expected_k)

        print(
            f"[STEP] variant={name} step={step} past_seen={past_seen} "
            f"seen={cache.get_seq_length(0)} physical={physical_lengths} "
            f"mask={mask_lengths} position={position_ids} scores={scores_lengths}"
        )
        print(f"[CHECK] variant={name} step={step} id=I1 expected={prompt_len + step - 1} actual={position_ids} verdict={'PASS' if i1 else 'FAIL'}")
        print(f"[CHECK] variant={name} step={step} id=I2 expected={physical_lengths} actual={mask_lengths} verdict={'PASS' if i2 else 'FAIL'}")
        print(f"[CHECK] variant={name} step={step} id=I3 expected_equal=True actual={physical_lengths} verdict={'PASS' if i3 else 'FAIL'}")
        print(f"[CHECK] variant={name} step={step} id=I4 expected={physical_lengths} actual={scores_lengths} verdict={'PASS' if i4 else 'FAIL'}")
        print(f"[CHECK] variant={name} step={step} id=I5 expected_original_k=True actual={i5} verdict={'PASS' if i5 else 'FAIL'}")
        if not all((i1, i2, i3, i4, i5)):
            print(f"[RESULT] variant={name} verdict=FAIL step={step}")
            manager.remove_hooks()
            for hook in hooks:
                hook.remove()
            return False

    manager.remove_hooks()
    for hook in hooks:
        hook.remove()
    print(f"[RESULT] variant={name} verdict=PASS steps={steps}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    print(f"[HDR] nonce={args.nonce}")
    print(f"[HDR] time={datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print(f"[HDR] python={sys.version.split()[0]}")
    print(f"[HDR] torch={torch.__version__}")
    print(f"[HDR] transformers={transformers.__version__}")
    print("[HDR] device=cpu")
    print("[HDR] seed=42")
    print(f"[HDR] script_sha256={sha256_file(__file__)}")
    torch.manual_seed(42)

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
    prompt_len = 64
    w_ols = torch.eye(16).view(1, 1, 16, 16).expand(2, 2, 16, 16).clone()

    h2o_model = AutoModelForCausalLM.from_config(config).eval()
    h2o_cache = H2OKVCache(compression_ratio=0.5, min_recent_tokens=8, num_sink_tokens=4, max_budget=32)
    h2o_ok = run_variant("h2o", h2o_cache, H2OAttentionHookManager, h2o_model, prompt_len, args.steps)

    vector_model = AutoModelForCausalLM.from_config(config).eval()
    vector_cache = VectorKVCache(
        compression_ratio=0.5,
        approx_ratio=0.1,
        min_recent_tokens=8,
        num_sink_tokens=4,
        max_budget=32,
        w_ols=w_ols,
    )
    vector_ok = run_variant("vector", vector_cache, VectorAttentionHookManager, vector_model, prompt_len, args.steps)
    result = h2o_ok and vector_ok
    print(f"[RESULT] overall={'PASS' if result else 'FAIL'} h2o={h2o_ok} vector={vector_ok}")
    raise SystemExit(0 if result else 1)


if __name__ == "__main__":
    main()
