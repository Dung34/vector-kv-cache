"""
CLI Runner: Phase 5–6 Offline OLS Calibration for VECTOR.
Computes layer-wise Per-Head W_OLS matrices (N_layers x N_heads x d_head x d_head)
and evaluates R^2 predictability score on held-out validation samples.

Usage:
    python scripts/run_offline_calibration.py --model_id Qwen/Qwen2.5-1.5B-Instruct --num_samples 16 --seq_len 1024
"""

import argparse
import os
import sys
import time
import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.calibration.ols_solver import PerHeadOLSSolver
from src.calibration.collector import ActivationCollector
from src.utils.model_utils import load_model_and_tokenizer


SAMPLE_PARAGRAPHS = [
    "Artificial intelligence and large language models represent a transformative paradigm in modern computational science. "
    "By employing deep transformer architectures, these systems can learn intricate contextual representations across extensive textual domains. "
    "A central component of autoregressive generation is the attention mechanism, which computes pairwise similarity scores between queries and keys. "
    "As sequences scale to thousands of tokens, managing the intermediate states known as the KV cache becomes a critical engineering challenge. ",
    "The mathematical foundation of quantum mechanics describes physical phenomena at microscopic scales using wavefunctions and Hilbert spaces. "
    "Superposition and entanglement enable quantum computers to explore exponentially large state spaces simultaneously. "
    "Algorithms such as Shor's algorithm for prime factorization and Grover's search algorithm offer theoretical speedups over classical counterparts. "
    "However, maintaining physical quantum coherence in the presence of environmental noise necessitates sophisticated error-correcting codes. ",
    "In financial econometrics, empirical asset pricing models seek to explain the cross-section of expected stock returns. "
    "The Capital Asset Pricing Model posits that asset risk is captured solely by covariance with the market portfolio. "
    "Multi-factor models expand this framework by incorporating size, value, momentum, and profitability factors to account for observed market anomalies. "
    "High-frequency trading strategies further exploit microsecond-level order book dynamics and liquidity imbalances. ",
    "Distributed computing systems handle petabytes of unstructured data by distributing workloads across clusters of commodity hardware. "
    "Consensus protocols like Paxos and Raft guarantee consistency across replicated state machines even under network partitions. "
    "Modern cloud data warehouses utilize columnar storage, vectorized query execution, and decoupled compute-storage architectures to accelerate analytics. "
    "Resource orchestration systems dynamically schedule containerized microservices to maximize hardware utilization. ",
]


def generate_calibration_prompt(tokenizer, target_len: int, seed: int) -> str:
    """Generates synthetic calibration text of exact target token length."""
    torch.manual_seed(seed)
    repeats = (target_len // 40) + 2
    full_text = " ".join(SAMPLE_PARAGRAPHS * repeats)
    tokens = tokenizer.encode(full_text, add_special_tokens=False)[:target_len]
    return tokenizer.decode(tokens)


def main():
    parser = argparse.ArgumentParser(description="VECTOR Phase 5–6 Offline OLS Calibration")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of training sequences")
    parser.add_argument("--val_samples", type=int, default=4, help="Number of validation sequences")
    parser.add_argument("--seq_len", type=int, default=1024, help="Token length per sequence")
    parser.add_argument("--ridge_alpha", type=float, default=1e-4, help="L2 regularization parameter")
    parser.add_argument("--output_file", type=str, default="weights/w_ols_qwen2.5_1.5b.pt")
    parser.add_argument("--chart_file", type=str, default="results/calibration/r2_layer_chart.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.chart_file), exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("VECTOR PHASE 5–6: OFFLINE OLS CALIBRATION & RoPE DECOUPLING")
    print(f"Model: {args.model_id} | Device: {device}")
    print(f"Calibration: {args.num_samples} samples x {args.seq_len} tokens | Validation: {args.val_samples} samples")
    print(f"Ridge Regularization: lambda = {args.ridge_alpha}")
    print("=" * 70)

    # 1. Load Model & Tokenizer
    model, tokenizer = load_model_and_tokenizer(model_id=args.model_id, device=device)
    config = model.config
    num_layers = getattr(config, "num_hidden_layers", 28)
    num_kv_heads = getattr(config, "num_key_value_heads", 2)
    hidden_size = getattr(config, "hidden_size", 1536)
    head_dim = getattr(config, "head_dim", hidden_size // getattr(config, "num_attention_heads", 12))

    print(f"Config: {num_layers} layers | {num_kv_heads} KV heads per layer | Head dim: {head_dim}")

    # 2. Initialize Solver & Collector
    solver = PerHeadOLSSolver(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        ridge_alpha=args.ridge_alpha,
        device=device,
        dtype=torch.float32,
    )
    collector = ActivationCollector(
        model=model,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
    )

    # 3. Calibration Phase: Accumulate Covariances
    print(f"\n[Step 1/3] Accumulating covariance statistics across {args.num_samples} calibration samples...")
    t0 = time.time()
    collector.register_hooks()

    for idx in range(args.num_samples):
        prompt = generate_calibration_prompt(tokenizer, args.seq_len, seed=1000 + idx)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            _ = model(**inputs)

        # Feed captured K, V pairs into OLS solver
        for layer_idx in range(num_layers):
            k, v = collector.get_layer_kv(layer_idx)
            if k is not None and v is not None:
                solver.update_batch(layer_idx, k, v)
            del k, v

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  Processed sample {idx + 1:2d}/{args.num_samples} ({(idx + 1) * args.seq_len:,} tokens accumulated)")

    collector.remove_hooks()
    t_calib = time.time() - t0
    print(f"[Done] Calibration completed in {t_calib:.1f}s. Total tokens: {solver.total_tokens:,}")

    # 4. Solve Closed-Form OLS Equations
    print("\n[Step 2/3] Solving closed-form normal equations W_OLS = Sigma_VK * (Sigma_KK + lambda*I)^(-1)...")
    w_ols = solver.solve()
    print(f"  Solved W_OLS weight tensor shape: {tuple(w_ols.shape)} ({w_ols.numel() * 4 / (1024**2):.2f} MB)")

    # 5. Validation Phase: Evaluate R^2 on Held-Out Samples
    print(f"\n[Step 3/3] Evaluating R^2 coefficient of determination on {args.val_samples} held-out validation samples...")
    collector.register_hooks()
    layer_r2_scores = {l: [] for l in range(num_layers)}

    for idx in range(args.val_samples):
        prompt = generate_calibration_prompt(tokenizer, args.seq_len, seed=9000 + idx)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            _ = model(**inputs)

        for layer_idx in range(num_layers):
            k_val, v_val = collector.get_layer_kv(layer_idx)
            if k_val is not None and v_val is not None:
                r2_metrics = solver.compute_r2(layer_idx, k_val, v_val)
                layer_r2_scores[layer_idx].append(r2_metrics["mean_r2"])
            del k_val, v_val

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    collector.remove_hooks()

    # Aggregate R^2 per layer
    layer_avg_r2 = [float(np.mean(layer_r2_scores[l])) for l in range(num_layers)]
    global_mean_r2 = float(np.mean(layer_avg_r2))

    print("\n" + "=" * 65)
    print(f"{'Layer':^8} | {'Per-Head W_OLS Shape':^22} | {'Validation R^2':^15} | {'Quality':^12}")
    print("-" * 65)
    for l in range(num_layers):
        r2 = layer_avg_r2[l]
        quality = "EXCELLENT" if r2 >= 0.85 else ("GOOD" if r2 >= 0.70 else "MODERATE")
        print(f"Layer {l:2d}  | {f'[{num_kv_heads}, {head_dim}, {head_dim}]':^22} | {r2 * 100:13.2f}% | {quality:^12}")
    print("-" * 65)
    print(f"GLOBAL AVERAGE R^2: {global_mean_r2 * 100:.2f}% (Target: >= 85.0%)")
    print("=" * 65)

    # 6. Save Weights Checkpoint
    checkpoint = {
        "w_ols": w_ols.cpu(),
        "model_id": args.model_id,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "global_mean_r2": global_mean_r2,
        "layer_r2": layer_avg_r2,
        "ridge_alpha": args.ridge_alpha,
        "calibrated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(checkpoint, args.output_file)
    print(f"\n[SAVED] Checkpoint saved successfully to: {args.output_file}")

    # 7. Plot R^2 Bar Chart
    plt.figure(figsize=(12, 5))
    colors = ["#2ecc71" if r >= 0.85 else ("#f39c12" if r >= 0.70 else "#e74c3c") for r in layer_avg_r2]
    bars = plt.bar(range(num_layers), [r * 100 for r in layer_avg_r2], color=colors, edgecolor="black", linewidth=0.5)
    plt.axhline(global_mean_r2 * 100, color="#2c3e50", linestyle="--", linewidth=1.5, label=f"Global Mean R^2: {global_mean_r2*100:.1f}%")
    plt.axhline(85.0, color="#27ae60", linestyle=":", linewidth=1.2, label="Target Threshold (85.0%)")

    plt.title(f"VECTOR Offline OLS Calibration: K->V Predictability (R^2) per Layer\nModel: {args.model_id.split('/')[-1]}", fontsize=12, fontweight="bold")
    plt.xlabel("Transformer Layer Index", fontsize=11)
    plt.ylabel("R^2 Score (%)", fontsize=11)
    plt.ylim(0, 105)
    plt.xticks(range(num_layers), fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(args.chart_file, dpi=300)
    plt.close()
    print(f"[SAVED] Validation chart saved to: {args.chart_file}")


if __name__ == "__main__":
    main()
