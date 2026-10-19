# Tasks Checklist: Review & Optimization of VECTOR KV Cache

## Phase 1: Algorithm Fixes & Throughput Optimization
- [x] Task 1: Add Attention Sink (Initial Tokens) to H2OKVCache and VectorKVCache <!-- id: 1 -->
  - Body: Retain fixed 4-16 initial tokens without eviction to stabilize attention distribution.
  - Acceptance Criteria: Both H2O and VECTOR preserve initial tokens; perplexity does not explode.
  - Verification: `python -m unittest tests.test_h2o_cache tests.test_vector_cache` passes.
- [x] Task 2: Vectorize Partitioning and Cache Reconstruction Error <!-- id: 2 -->
  - Body: Eliminate Python head-loops with `torch.nonzero()`, cache $\epsilon_i$ per token so it is evaluated only once on arrival.
  - Acceptance Criteria: Throughput on decode step increases significantly (>20-30 tokens/s on GPU).
  - Verification: Latency timer benchmark shows speedup > 5x.

## Checkpoint: Core Stability
- [x] Attention distribution verified stable
- [x] Throughput benchmark exceeds 20 tokens/sec

## Phase 2: Core Empirical Verification
- [ ] Task 3: Re-evaluate Single-Needle NIAH Preview on T4 GPU <!-- id: 3 -->
  - Body: Test needle retrieval at 50% depth on 2048/4096 tokens with p_c=75%.
  - Acceptance Criteria: Model successfully retrieves target needle, resolving previous preview failure.
  - Verification: Script `scripts/run_niah_benchmark.py` outputs PASS for needle key.
- [ ] Task 4: Address `output_attentions=True` Memory Overhead <!-- id: 4 -->
  - Body: Prevent massive attention matrix allocation during long-context prefill.
  - Acceptance Criteria: MemoryTracker confirms no multi-GB spike during prefill at 8K tokens.
  - Verification: Memory benchmark on 4K/8K tokens runs without OOM.

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
