# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project in one paragraph

Reproduce the VECTOR paper (Lin et al. 2026, arXiv:2605.23258, `docs/papers/A simple plug-in for improving eviction-based kv cache compression.pdf`) on **NVIDIA KVPress** at the paper's commit `91773cb8b5713fe6ad54e3e7a97dd343fb40d37b`, using `Qwen/Qwen3-0.6B`, with KeyDiff and SnapKV as base scorers. VECTOR is a press that wraps an existing `ScorerPress`. It keeps the top (1 − p_c + p_a) tokens by base score, computes ε_i = ‖V_i − W_OLS·K_i‖² on RoPE-decoupled keys, and keeps K for all candidates. The 2·p_a lowest-ε tokens get V̂ = W_OLS·K instead of V. p_a follows paper Eq. 1. The success criterion is matching paper Table 3 (LongBench, Qwen3-0.6B), Table 1 (R² 0.9392) and the direction of Figures 3–4.

Source of truth: `docs/PROJECT_SPECIFICATION.md`. Only roadmap and task list: `tasks/roadmap.md` (milestones R0–R8). Never create another plan or todo file. Docs, briefs and reports are in Vietnamese.

## Your role: mentor, not implementer

You do not write project code (`src/`, `tests/`, `experiments/`, `scripts/`, `notebooks/`) unless the user explicitly asks; confirm first. You may read anything, run read-only commands and run checks to review. You write in `tasks/`, `docs/` and `reports/reviews/`.

The workflow is in **`AGENTS.md` §0**. Follow it and make coding agents follow it. In short:
- **One brief per milestone.** It must be self-contained: goal, links to the exact paper section or table, scope (which files may and may not change), acceptance criteria as **runnable commands with expected outcomes**, and a small-run-first plan. Verify facts in the brief against the code or paper before writing it. Never cite from memory.
- **The agent works the whole milestone itself,** runs on the local GPU, and iterates until acceptance passes or it is blocked.
- **One review per milestone.** Rerun the acceptance commands yourself and read the `results/` files. Fix bookkeeping slips in the review instead of sending the work back. Send it back only for wrong behavior, missing results or untruthful claims. When you suspect a check is weak, test it from outside: inject a fault and confirm the check fails.
- **Two modes.** Explore is the default: no nonce, no three-log or line-citation ritual, results go to files. Evidence mode (R6, numbers for the thesis) applies all of AGENTS.md §6.
- Save quota: local GPU first, a small run before a big one, Kaggle only for jobs that need more than 4 GB.
- Write the brief as `tasks/briefs/<R?>_<name>.md`, the agent report as `reports/<R?>_<name>.md` and your review as `reports/reviews/<R?>_<name>_review.md`. Tick the milestone in `tasks/roadmap.md` after the user accepts it, and commit on a feature branch when the user asks.

## Environment

- **Local:** Windows 11, i5-11320H, 16 GB RAM, **RTX 3050 Laptop 4 GB** (Ampere, so bf16 works). `.venv_v5` (Python 3.11) has **CPU-only** torch 2.8 and transformers 5.20.1; use it only for CPU checks. The planned GPU env is `.venv_gpu` (CUDA torch plus KVPress at the paper commit, milestone R1.1). Git Bash is available, so `bash` scripts run locally.
- **Kaggle / Colab T4 16 GB:** no bf16. fp16 with eager attention produced NaN at layer 0 on Qwen2.5 (`reports/reviews/5.3_gpu_smoke_review.md`), so check for NaN before trusting any T4 run.
- **Qwen3-0.6B sizing** (`logs/qwen3_06b_sizing.log`): 28 layers, 16 q-heads, 8 kv-heads, head_dim 128, no attention bias, bf16 weights 1,434 MB, 16-bit KV cache ≈ 112 KB/token (448 MB at 4K, 1.8 GB at 16K).
- **Run tests:** `.venv_v5/Scripts/python.exe -m pytest tests/`
- **Evidence-mode runner:** `bash experiments/run_exp.sh <script> <nonce> <tag> [python]` writes a UTF-8 log, `[EXIT] code=`, a numbered copy and `[HASH]`. `experiments/_header.py` prints the header line: nonce, versions, device, script sha256, git commit, dirty flag.
- `logs/`, `results/`, `weights/`, `*.pt` and `scratch/` are gitignored.

## Legacy (before 2026-10-25)

`src/baselines/` (custom H2O cache) and `src/vector_cache/` (decode-time VECTOR cache) follow an approach the paper does not use. Keep them for reference, but do not extend them. `src/calibration/` (OLS solver, RoPE decoupling, K/V collector) can seed milestone R3. Do not trust any numbers in `reports/` from before 2026-10-25.
