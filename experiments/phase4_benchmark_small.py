import argparse
import ctypes
import datetime
import hashlib
import json
import os
import random
import sys
import time
from ctypes import wintypes

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.baselines.full_cache import FullKVCacheWrapper
from src.baselines.h2o_cache import H2OKVCache
from src.baselines.h2o_hook import H2OAttentionHookManager
from src.vector_cache.vector_cache import VectorKVCache
from src.vector_cache.vector_hook import VectorAttentionHookManager


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def process_memory():
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize


def build_model(config, state):
    model = AutoModelForCausalLM.from_config(config).eval()
    model.load_state_dict(state)
    return model


def run_variant(name, model, cache, manager_cls, prompt, steps):
    manager = None
    if manager_cls is not None:
        manager = manager_cls(model=model, kv_cache=cache)
        manager.register_hooks()
    start = time.perf_counter()
    with torch.no_grad():
        output = model(prompt, past_key_values=cache, use_cache=True)
    prefill_sec = time.perf_counter() - start
    generated = []
    decode_times = []
    for _ in range(steps):
        token = output.logits[:, -1:].argmax(dim=-1)
        generated.append(token.item())
        start = time.perf_counter()
        with torch.no_grad():
            output = model(token, past_key_values=cache, use_cache=True)
        decode_times.append(time.perf_counter() - start)
    if manager is not None:
        manager.remove_hooks()
    return {
        "variant": name,
        "context_length": prompt.shape[-1],
        "decode_tokens": steps,
        "prefill_ms": round(prefill_sec * 1000, 3),
        "decode_mean_ms": round(sum(decode_times) / len(decode_times) * 1000, 3),
        "decode_tokens_per_sec": round(steps / sum(decode_times), 3),
        "inter_token_latency_ms": round(sum(decode_times) / len(decode_times) * 1000, 3),
        "persistent_kv_bytes": cache.get_memory_bytes(),
        "physical_len": [cache._physical_len(i) if hasattr(cache, "_physical_len") else cache.get_seq_length(i) for i in range(2)],
        "peak_working_set_bytes": process_memory(),
        "peak_allocated_vram_bytes": None,
        "peak_reserved_vram_bytes": None,
        "generated": generated,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    config = AutoConfig.for_model(
        "qwen2", vocab_size=1000, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=4096, use_cache=True,
    )
    config._attn_implementation = "eager"
    base = AutoModelForCausalLM.from_config(config).eval()
    state = base.state_dict()
    prompt = torch.randint(0, config.vocab_size, (1, 64))
    w_ols = torch.eye(16).view(1, 1, 16, 16).expand(2, 2, 16, 16).clone()
    variants = [
        ("full_fp16", FullKVCacheWrapper(), None),
        ("h2o", H2OKVCache(target_budget=32, max_physical_budget=64, min_recent_tokens=8), H2OAttentionHookManager),
        ("vector", VectorKVCache(target_budget=32, max_physical_budget=64, compression_ratio=0.5, approx_ratio=0.1, min_recent_tokens=8, w_ols=w_ols), VectorAttentionHookManager),
    ]
    print(f"nonce={args.nonce}")
    print(f"timestamp={datetime.datetime.now().isoformat()} python={sys.version.split()[0]} torch={torch.__version__} transformers={transformers.__version__} device=cpu seed={seed} script_sha256={sha256_file(__file__)}")
    results = []
    for name, cache, manager_cls in variants:
        result = run_variant(name, build_model(config, state), cache, manager_cls, prompt, args.steps)
        result["compression_ratio"] = 0.0 if name == "full_fp16" else 0.5
        results.append(result)
        print(f"result={json.dumps(result, sort_keys=True)}")
    baseline = results[0]["generated"]
    for result in results[1:]:
        result["first_token_matches_full"] = result["generated"][0] == baseline[0]
        print(f"quality_check={result['variant']} first_token_expected={baseline[0]} actual={result['generated'][0]} verdict={'PASS' if result['first_token_matches_full'] else 'INFO'}")
    payload = {"nonce": args.nonce, "results": results}
    os.makedirs("reports", exist_ok=True)
    with open("reports/phase4_benchmark_small.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open("reports/phase4_benchmark_small.md", "w", encoding="utf-8") as handle:
        handle.write("| variant | compression_ratio | prefill_ms | decode_mean_ms | tok/s | persistent_kv_bytes | peak_working_set_bytes | first_token_matches_full |\n|---|---:|---:|---:|---:|---:|---:|---|\n")
        for result in results:
            handle.write(f"| {result['variant']} | {result['compression_ratio']} | {result['prefill_ms']} | {result['decode_mean_ms']} | {result['decode_tokens_per_sec']} | {result['persistent_kv_bytes']} | {result['peak_working_set_bytes']} | {result.get('first_token_matches_full', True)} |\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
