"""First GPU infrastructure smoke test for the full KV-cache baseline."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def _torch() -> Any:
    import torch

    return torch


def _ensure_repo_on_path() -> None:
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _print_header(nonce: str) -> None:
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            from experiments._header import print_header
        except ModuleNotFoundError:
            from _header import print_header
    print_header(nonce, seed=42, script_path=__file__)


def build_model(mode: str) -> Tuple[Any, Optional[Any]]:
    torch = _torch()
    if mode == "gpu":
        _ensure_repo_on_path()
        from src.utils.model_utils import load_model_and_tokenizer

        return load_model_and_tokenizer(
            "Qwen/Qwen2.5-1.5B-Instruct",
            device="cuda",
            attn_implementation="eager",
        )

    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    torch.manual_seed(42)
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
    return Qwen2ForCausalLM(config).eval(), None


def make_prompt(mode: str, tokenizer: Optional[Any], model: torch.nn.Module) -> torch.Tensor:
    torch = _torch()
    device = next(model.parameters()).device
    if mode == "gpu":
        encoded = tokenizer("The quick brown fox tests the cache.", return_tensors="pt")
        return encoded.input_ids.to(device)
    return torch.randint(0, 1000, (1, 16), device=device)


def run_one_step(
    model: Any,
    input_ids: torch.Tensor,
    cache: Any,
    decode_token_delta: int = 0,
) -> Dict[str, Any]:
    torch = _torch()
    with torch.no_grad():
        prefill = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        next_token = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
        decode_token = next_token
        if decode_token_delta:
            decode_token = (next_token + decode_token_delta) % model.config.vocab_size
        decode = model(input_ids=decode_token, past_key_values=cache, use_cache=True)
    return {
        "prefill_logits": prefill.logits,
        "decode_logits": decode.logits,
        "next_token": next_token,
        "decode_token": decode_token,
        "cache": cache,
    }


def _check(name: str, expected: Any, actual: Any) -> Dict[str, Any]:
    ok = expected == actual
    print(f"[CHECK] name={name} expected={expected} actual={actual} ok={ok}")
    return {"name": name, "expected": expected, "actual": actual, "ok": ok}


def run_checks(model: torch.nn.Module, input_ids: torch.Tensor) -> List[Dict[str, Any]]:
    torch = _torch()
    _ensure_repo_on_path()
    from src.baselines.full_cache import FullKVCacheWrapper
    from transformers.cache_utils import DynamicCache

    full = run_one_step(model, input_ids, FullKVCacheWrapper())
    dynamic = run_one_step(model, input_ids, DynamicCache())
    full_cache = full["cache"]
    finite = bool(
        torch.isfinite(full["prefill_logits"]).all()
        and torch.isfinite(full["decode_logits"]).all()
    )
    checks = [_check("logits_finite", True, finite)]

    prompt_len = input_ids.shape[-1]
    seq_len = full_cache.get_seq_length()
    physical_len = full_cache.key_cache[0].shape[-2]
    max_abs_diff = float(
        torch.max(torch.abs(full["decode_logits"] - dynamic["decode_logits"])).item()
    )
    print(
        f"[CACHE] next_token={int(full['next_token'].item())} "
        f"decode_token={int(full['decode_token'].item())} "
        f"decode_dynamic_token={int(dynamic['decode_token'].item())} "
        f"max_abs_diff={max_abs_diff:.8g} "
        f"seq_len={seq_len} physical_len_layer0={physical_len}"
    )
    checks.append(_check("seq_len_after_decode", prompt_len + 1, seq_len))

    expected_token = int(dynamic["decode_logits"][:, -1].argmax(dim=-1).item())
    actual_token = int(full["decode_logits"][:, -1].argmax(dim=-1).item())
    checks.append(_check("token_matches_dynamic_cache", expected_token, actual_token))
    checks.append(_check("decode_logits_match_dynamic_cache", True, max_abs_diff <= 1e-5))
    return checks


def _print_model_info(model: Any) -> None:
    torch = _torch()
    config = model.config
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    values = {
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": head_dim,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "dtype": str(next(model.parameters()).dtype),
        "attn_implementation": getattr(config, "_attn_implementation", "unknown"),
    }
    print("[MODEL] " + " ".join(f"{key}={value}" for key, value in values.items()))
    if torch.cuda.is_available():
        print(f"[MODEL] memory_allocated_after_load={torch.cuda.memory_allocated()}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", required=True)
    args = parser.parse_args(argv)
    _print_header(args.nonce)
    torch = _torch()

    mode = "smoke" if os.environ.get("PHASE5_SMOKE") == "1" else "gpu"
    print(f"[MODE] mode={mode}")
    if mode == "gpu" and not torch.cuda.is_available():
        print("[FAIL] reason=cuda_not_available")
        return 1

    model, tokenizer = build_model(mode)
    _print_model_info(model)
    input_ids = make_prompt(mode, tokenizer, model)
    print(f"[PROMPT] prompt_len={input_ids.shape[-1]}")
    checks = run_checks(model, input_ids)
    all_ok = all(check["ok"] for check in checks)
    print(f"[RESULT] all_ok={str(all_ok).lower()}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
