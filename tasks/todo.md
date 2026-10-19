# Tasks Checklist: Review & Optimization of VECTOR KV Cache

## Phase 1: Algorithm Fixes & Throughput Optimization
- [x] Task 1: Add Attention Sink (Initial Tokens) to H2OKVCache and VectorKVCache <!-- id: 1 -->
  - Body: Retain fixed 4-16 initial tokens without eviction to stabilize attention distribution.
  - Acceptance Criteria: Both H2O and VECTOR preserve initial tokens; perplexity does not explode.
  - Verification: `python -m unittest tests.test_h2o_cache tests.test_vector_cache` passes.
- [x] Task 1.3: Optimize Pruning with Pre-allocated Index Buffers & Redundant Sort Elimination <!-- id: 1.3 -->
  - Body: Cache sink and recent indices, eliminate full `torch.sort` on concatenated indices in `h2o_cache.py` and `partitioner.py`.
  - Acceptance Criteria: Pruning passes all existing unit tests without allocating fresh `arange` buffers every step.
  - Verification: `python -m unittest tests.test_h2o_cache tests.test_partitioner` passes.
- [x] Task 1.4: Decouple Prefill ($T_{\text{prefill}}$ ms) and Decode ($T_{\text{decode}}$ tok/s) in Benchmarks <!-- id: 1.4 -->
  - Body: Implement clean benchmark routine measuring prefill latency (TTFT) and decode tokens/sec separately with CUDA synchronization.
  - Acceptance Criteria: Benchmark table explicitly outputs Prefill (ms) and Decode (tok/s).
  - Verification: Run benchmark cell and verify table output contains both decoupled metrics.

## Checkpoint: Core Stability
- [x] Attention distribution verified stable
- [x] Throughput benchmark exceeds 20 tokens/sec

## Phase 2: Core Empirical Verification (VECTOR vs Baselines)
- [x] Task 2.1: Upgrade and Standardize VECTOR Kaggle Notebook to v2.0 <!-- id: 2.1 -->
  - Body: Create `kaggle_phase7_8_vector_cache_v2.ipynb` integrating decoupled prefill/decode benchmark, Qwen2.5 Chat Template, and Multi-Needle NIAH (3 & 5 needles @ 4096 tokens).
  - Acceptance Criteria: Notebook runs end-to-end, evaluates Full vs H2O vs VECTOR, and generates automated academic report v2.0.
  - Verification: Run validation test on notebook code with mock forward passes.
- [ ] Task 2.2: Verify Budget Swap Invariance and Single-Needle Recovery <!-- id: 2.2 -->
  - Body: Verify VECTOR recovers needle accuracy at 50% depth (overcoming H2O's 0% blind spot) at p_c=75%.
  - Acceptance Criteria: VECTOR single-needle accuracy >= 90%.
  - Verification: Verification run passes and outputs passkey correctly.
- [ ] Task 2.3: Multi-Needle Recall Evaluation (3 & 5 Needles) <!-- id: 2.3 -->
  - Body: Benchmark multi-needle retrieval across Full FP16, H2O (50%, 75%), and VECTOR (75%).
  - Acceptance Criteria: VECTOR outperforms H2O 75% (which scored 0.0%) by >= 80% recall.
  - Verification: Output table records multi-needle recall metrics.

## Checkpoint: Empirical Success
- [ ] NIAH Single-Needle passes with high accuracy
- [ ] Peak VRAM during prefill remains within budget

## Phase 3: Comprehensive Benchmark & Reporting
- [ ] Task 5: Execute Multi-Needle NIAH Benchmark <!-- id: 5 -->
  - Body: Run comparison between Full FP16, H2O, and VECTOR on 3-5 needles.
  - Acceptance Criteria: VECTOR outperforms H2O by >= 15% recall on multi-needle retrieval.
  - Verification: Generated charts and JSON reports saved in `results/`.
- [ ] Task 6: Update Experimental Reports & Final Documentation <!-- id: 6 -->
  - Body: Compile accurate metrics into markdown reports and Project III thesis draft.
  - Acceptance Criteria: Clean, academic report with comparative tables and Pareto charts.
  - Verification: Report markdown files verified and rendered cleanly.
