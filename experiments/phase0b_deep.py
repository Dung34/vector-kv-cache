import os
import sys
import argparse
import datetime
import hashlib
import inspect
import warnings

# Suppress PyTorch distributed / redirect warnings on Windows
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TORCH_LOGS"] = "-all"

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import transformers
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM
import transformers.models.qwen2.modeling_qwen2 as qm

def compute_file_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", type=str, required=True)
    args = parser.parse_args()

    script_path = os.path.abspath(__file__)
    script_sha256 = compute_file_sha256(script_path)

    # 1. In header [HDR]
    print(f"[HDR] nonce={args.nonce}")
    print(f"[HDR] time={datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print(f"[HDR] python={sys.version.split()[0]}")
    print(f"[HDR] torch={torch.__version__}")
    print(f"[HDR] transformers={transformers.__version__}")
    print(f"[HDR] device=cpu")
    print(f"[HDR] seed=42")
    print(f"[HDR] script_sha256={script_sha256}")

    # Version guard: dừng nếu không nằm trong {4.56.2, 5.20.1}
    SUPPORTED_VERSIONS = ("4.56.2", "5.20.1")
    if transformers.__version__ not in SUPPORTED_VERSIONS:
        print(f"[ERR] type=RuntimeError msg=Unsupported transformers version {transformers.__version__}")
        print(f"[RESULT] FAIL reason=unsupported_version")
        sys.exit(1)

    torch.manual_seed(42)

    # =========================================================================
    # ITEM 1 & 2: PATCH MECHANISM & ID CHECK (TRƯỚC / SAU PATCH)
    # =========================================================================
    fn_before = qm.eager_attention_forward
    id_before = id(fn_before)
    name_before = getattr(fn_before, "__qualname__", str(fn_before))
    print(f"[CFG] before_patch_fn={name_before} id={id_before}")

    from src.utils.mask_patch import (
        patch_hf_eager_attention_for_pruned_cache,
        get_slice_count,
        reset_slice_count,
    )
    reset_slice_count()
    patch_hf_eager_attention_for_pruned_cache()

    fn_after = qm.eager_attention_forward
    id_after = id(fn_after)
    name_after = getattr(fn_after, "__qualname__", str(fn_after))
    print(f"[CFG] after_patch_fn={name_after} id={id_after}")
    print(f"[CHECK] id=PATCH_HOOK expected=True actual={id_before != id_after} verdict={'PASS' if id_before != id_after else 'FAIL'}")

    if hasattr(qm.ALL_ATTENTION_FUNCTIONS, "get_interface"):
        interface_fn = qm.ALL_ATTENTION_FUNCTIONS.get_interface("eager", qm.eager_attention_forward)
    else:
        interface_fn = getattr(qm, "eager_attention_forward", None)
    print(f"[CFG] interface_fn={getattr(interface_fn, '__qualname__', str(interface_fn))} id={id(interface_fn)}")

    # =========================================================================
    # ITEM 2: IN SOURCE ĐẦY ĐỦ CỦA H2OKVCache.get_mask_sizes
    # =========================================================================
    from src.baselines.h2o_cache import H2OKVCache
    lines, start_line = inspect.getsourcelines(H2OKVCache.get_mask_sizes)
    source_file = inspect.getsourcefile(H2OKVCache.get_mask_sizes)
    print(f"[SRC] file={source_file} line={start_line}")
    for l in lines:
        print(f"[SRC] {l.rstrip()}")

    # =========================================================================
    # ITEM 3: MRO, CACHE.LAYERS[I].KEYS VÀ DATA_PTR SO VỚI CACHE._KEY_CACHE[I]
    # =========================================================================
    mro_names = [f"[{i}] {cls.__module__}.{cls.__qualname__}" for i, cls in enumerate(H2OKVCache.__mro__)]
    for entry in mro_names:
        print(f"[CFG] mro={entry}")

    cache = H2OKVCache(compression_ratio=0.5, min_recent_tokens=8, num_sink_tokens=4)

    config = Qwen2Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        _attn_implementation="eager"
    )
    model = Qwen2ForCausalLM(config).eval()

    prefill_len = 64
    prompt = torch.randint(0, 1000, (1, prefill_len))
    cache.is_prefilling = True
    with torch.no_grad():
        model(prompt, past_key_values=cache, use_cache=True, output_attentions=True)
    cache.is_prefilling = False

    print(f"[CFG] prefill_len={prefill_len}")
    print(f"[CFG] cache_has_layers={hasattr(cache, 'layers')}")
    print(f"[CFG] cache_layers_len={len(cache.layers) if hasattr(cache, 'layers') else 0}")
    print(f"[CFG] cache_key_cache_len={len(cache._key_cache)}")

    # So sánh keys và data_ptr từng layer
    for i in range(2):
        layer_i = cache.layers[i] if hasattr(cache, "layers") and i < len(cache.layers) else None
        layer_has_keys = hasattr(layer_i, "keys") and layer_i.keys is not None
        layer_keys_shape = tuple(layer_i.keys.shape) if layer_has_keys else None
        k_cache_shape = tuple(cache._key_cache[i].shape) if i < len(cache._key_cache) else None
        same_ptr = False
        if layer_has_keys and i < len(cache._key_cache):
            same_ptr = (layer_i.keys.data_ptr() == cache._key_cache[i].data_ptr())
        print(f"[CFG] layer={i} keys_shape={layer_keys_shape} key_cache_shape={k_cache_shape} same_data_ptr={same_ptr}")
        print(f"[CHECK] id=DATA_PTR_L{i} expected=True actual={same_ptr} verdict={'PASS' if same_ptr else 'FAIL'}")

    # =========================================================================
    # ITEM 4 & DECODE RUN: ĐẾM SỐ LẦN CẮT MASK KHI PATCH BẬT
    # =========================================================================
    step_records = {}
    def make_pre_hook(layer_idx):
        def hook_fn(module, args, kwargs):
            attn_mask = kwargs.get("attention_mask")
            step_records[f"layer_{layer_idx}_posids"] = kwargs.get("position_ids")
            step_records[f"layer_{layer_idx}_mask"] = attn_mask
            step_records[f"layer_{layer_idx}_mask_len"] = attn_mask.shape[-1] if attn_mask is not None else 0
        return hook_fn

    hooks = [
        model.model.layers[i].register_forward_pre_hook(make_pre_hook(i), with_kwargs=True)
        for i in range(2)
    ]

    expected_seen = prefill_len
    history = []

    for step in [1, 2]:
        step_records.clear()
        past_seen = cache.get_seq_length(0)
        token = torch.randint(0, 1000, (1, 1))

        with torch.no_grad():
            model(token, past_key_values=cache, use_cache=True, output_attentions=True)

        phys_len = cache.key_cache[0].shape[-2]
        scores_len = cache.accumulated_scores[0].shape[-1] if len(cache.accumulated_scores) > 0 and cache.accumulated_scores[0] is not None else 0
        pos_id_tensor = step_records.get("layer_0_posids")
        pos_id = pos_id_tensor[0, -1].item() if pos_id_tensor is not None and pos_id_tensor.numel() > 0 else -1
        mask_len = step_records.get("layer_0_mask_len", 0)

        for l in range(2):
            m = step_records.get(f"layer_{l}_mask")
            m_shape = tuple(m.shape) if m is not None else None
            m_min = m.min().item() if m is not None else float("nan")
            m_max = m.max().item() if m is not None else float("nan")
            print(f"[MASK] step={step} layer={l} shape={m_shape} min={m_min} max={m_max}")

        print(f"[STEP] n={step} seen={expected_seen} past_seen={past_seen} phys_len={phys_len} mask_kv={mask_len} pos_id={pos_id} scores_len={scores_len}")

        if step == 1:
            # Kiểm tra sau bước decode 1: layers[i].keys.shape, data_ptr so với _key_cache[i]
            for i in range(2):
                l_i = cache.layers[i] if hasattr(cache, "layers") and i < len(cache.layers) else None
                l_has_k = hasattr(l_i, "keys") and l_i.keys is not None
                l_k_shape = tuple(l_i.keys.shape) if l_has_k else None
                k_c_shape = tuple(cache._key_cache[i].shape) if i < len(cache._key_cache) else None
                s_ptr = False
                if l_has_k and i < len(cache._key_cache):
                    s_ptr = (l_i.keys.data_ptr() == cache._key_cache[i].data_ptr())
                two_copies = not s_ptr
                print(f"[CFG] post_decode1_layer={i} keys_shape={l_k_shape} key_cache_shape={k_c_shape} same_data_ptr={s_ptr} two_copies={two_copies}")
                print(f"[CHECK] id=POST_DEC1_PTR_L{i} expected=True actual={s_ptr} verdict={'PASS' if s_ptr else 'FAIL'}")
                print(f"[CHECK] id=POST_DEC1_TWO_COPIES_L{i} expected=False actual={two_copies} verdict={'PASS' if not two_copies else 'FAIL'}")
            # Đo process memory
            try:
                import psutil
                rss_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
                print(f"[CFG] process_memory_rss_mb={rss_mb}")
            except Exception as e:
                print(f"[ERR] type={type(e).__name__} msg={e}")

        history.append({
            "step": step,
            "expected_seen": expected_seen,
            "pos_id": pos_id,
            "phys_len": phys_len,
            "mask_len": mask_len,
        })
        expected_seen += 1

    for h in hooks:
        h.remove()

    slice_count_val = get_slice_count()
    print(f"[CFG] patch_slice_count={slice_count_val}")

    # Kiểm tra bất biến I1 và I2 ở bước 2
    step2 = history[1]
    verdict_i1 = "PASS" if step2["pos_id"] == step2["expected_seen"] else "FAIL"
    verdict_i2 = "PASS" if step2["mask_len"] == step2["phys_len"] else "FAIL"
    print(f"[CHECK] id=I1 expected={step2['expected_seen']} actual={step2['pos_id']} verdict={verdict_i1}")
    print(f"[CHECK] id=I2 expected={step2['phys_len']} actual={step2['mask_len']} verdict={verdict_i2}")

    # =========================================================================
    # ITEM 5: TÍNH TOÁN CÔNG THỨC TRÊN SỐ GIẢ THUYẾT 3755
    # =========================================================================
    h_cache = H2OKVCache(compression_ratio=0.50, min_recent_tokens=32, num_sink_tokens=4)
    h_prompt_len = 3755
    mock_k = torch.zeros(1, 2, h_prompt_len, 16)
    mock_v = torch.zeros(1, 2, h_prompt_len, 16)
    h_cache.is_prefilling = True
    h_cache.update(mock_k, mock_v, 0)
    h_cache.is_prefilling = False

    active_budget_val = h_cache.active_budget
    mask_sizes_val = h_cache.get_mask_sizes(1, 0)
    expected_mask_kv = mask_sizes_val[0]

    print(f"[CFG] formula_calc_prompt_len={h_prompt_len}")
    print(f"[CFG] formula_calc_active_budget={active_budget_val}")
    print(f"[CFG] formula_calc_mask_kv_len={expected_mask_kv}")
    check_hyp_budget = (active_budget_val == 1877)
    check_hyp_mask = (expected_mask_kv == 3756)
    print(f"[CHECK] id=FORMULA_BUDGET expected=1877 actual={active_budget_val} verdict={'PASS' if check_hyp_budget else 'FAIL'}")
    print(f"[CHECK] id=FORMULA_MASK expected=3756 actual={expected_mask_kv} verdict={'PASS' if check_hyp_mask else 'FAIL'}")

    if verdict_i1 == "FAIL":
        print(f"[RESULT] FAIL reason=I1_violated_step2_posid_{step2['pos_id']}_vs_{step2['expected_seen']}")
        sys.exit(1)
    else:
        print(f"[RESULT] PASS reason=all_checks_passed")

if __name__ == "__main__":
    main()
