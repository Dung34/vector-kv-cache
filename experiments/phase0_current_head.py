import argparse
import datetime
import hashlib
import inspect
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import transformers
import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model

from src.baselines.h2o_cache import H2OKVCache


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_source(label, function):
    source_file = inspect.getsourcefile(function)
    source_lines, start_line = inspect.getsourcelines(function)
    print(f"[SRC] label={label} file={source_file} line={start_line}")
    for offset, line in enumerate(source_lines):
        print(f"[SRC] {start_line + offset}: {line.rstrip()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", required=True)
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

    print_source("Qwen2Model.forward", Qwen2Model.forward)
    print_source("DynamicCache.get_seq_length", DynamicCache.get_seq_length)
    if hasattr(DynamicCache, "get_mask_sizes"):
        print_source("DynamicCache.get_mask_sizes", DynamicCache.get_mask_sizes)
    else:
        print("[SRC] label=DynamicCache.get_mask_sizes unavailable=True")
    print(f"[CFG] qwen2_all_attention_functions={list(qwen2_mod.ALL_ATTENTION_FUNCTIONS.keys())}")

    config = Qwen2Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        _attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(config).eval()
    cache = H2OKVCache(
        compression_ratio=0.5,
        min_recent_tokens=8,
        num_sink_tokens=4,
        max_budget=32,
    )

    call_log = []
    original_get_seq_length = cache.get_seq_length
    original_get_mask_sizes = cache.get_mask_sizes
    original_update = cache.update

    def logged_get_seq_length(*args, **kwargs):
        result = original_get_seq_length(*args, **kwargs)
        call_log.append(("get_seq_length", args, kwargs, result))
        return result

    def logged_get_mask_sizes(*args, **kwargs):
        result = original_get_mask_sizes(*args, **kwargs)
        call_log.append(("get_mask_sizes", args, kwargs, result))
        return result

    def logged_update(key_states, value_states, layer_idx, cache_kwargs=None):
        before = cache._physical_len(layer_idx)
        result = original_update(key_states, value_states, layer_idx, cache_kwargs)
        after = cache._physical_len(layer_idx)
        print(f"[UPDATE] layer={layer_idx} incoming_len={key_states.shape[-2]} physical_before={before} physical_after={after}")
        return result

    cache.get_seq_length = logged_get_seq_length
    cache.get_mask_sizes = logged_get_mask_sizes
    cache.update = logged_update

    layer_records = {}

    def make_layer_hook(layer_idx):
        def hook(module, args, kwargs):
            mask = kwargs.get("attention_mask")
            positions = kwargs.get("position_ids")
            layer_records[layer_idx] = {
                "mask_shape": tuple(mask.shape) if mask is not None else None,
                "position_shape": tuple(positions.shape) if positions is not None else None,
                "position_last": positions[0, -1].item() if positions is not None else None,
            }
        return hook

    hooks = [model.model.layers[i].register_forward_pre_hook(make_layer_hook(i), with_kwargs=True) for i in range(2)]

    prompt_len = 64
    prompt = torch.randint(0, config.vocab_size, (1, prompt_len))
    with torch.no_grad():
        model(prompt, past_key_values=cache, use_cache=True, output_attentions=True)
    print(f"[PREFILL] prompt_len={prompt_len} seen={cache.get_seq_length(0)} physical={cache._physical_len(0)} active_budget={cache.active_budget}")
    for layer_idx in sorted(layer_records):
        print(f"[LAYER] phase=prefill layer={layer_idx} data={layer_records[layer_idx]}")

    layer_records.clear()
    past_seen = cache.get_seq_length(0)
    token = torch.randint(0, config.vocab_size, (1, 1))
    try:
        with torch.no_grad():
            model(token, past_key_values=cache, use_cache=True, output_attentions=True)
        decode_error = None
    except Exception as exc:
        decode_error = f"{type(exc).__name__}: {exc}"
        print(f"[DECODE_ERROR] {decode_error}")

    print(f"[DECODE] past_seen={past_seen} physical_after={cache._physical_len(0)} seen_after={cache.get_seq_length(0)}")
    for layer_idx in sorted(layer_records):
        print(f"[LAYER] phase=decode layer={layer_idx} data={layer_records[layer_idx]}")
    for entry in call_log:
        print(f"[CACHE_CALL] name={entry[0]} args={entry[1]} kwargs={entry[2]} result={entry[3]}")

    for hook in hooks:
        hook.remove()

    print(f"[RESULT] decode_exception={decode_error is not None} layers_observed={len(layer_records)}")


if __name__ == "__main__":
    main()
