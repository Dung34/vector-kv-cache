# Lộ trình dự án — bản duy nhất

> **Đây là lộ trình và danh sách task DUY NHẤT của dự án.** Mọi brief (`tasks/briefs/<ID>_<tên>.md`) phải lấy ID từ file này.
> Ngày chốt: 2026-10-24. Thay thế và đã xóa `tasks/plan.md`, `tasks/todo.md`, `tasks/fixh2ovector.md` (xem lại bằng `git log -- tasks/`) và bản nháp `tasks/roadmap_v2.md` (nội dung đã gộp vào đây).
> Nguồn chuẩn: `docs/PROJECT_SPECIFICATION.md` (Spec). Nhận xét kiến trúc: `reports/reviews/architecture_review_20261024.md`.

## 0. Ràng buộc hạ tầng

| Nơi chạy | Dùng cho |
|---|---|
| 💻 **Local CPU** (`.venv_v5`, torch 2.8.0+cpu, transformers 5.20.1) | unit test, Qwen2 config nhỏ, probe, refactor |
| 🟢 **Kaggle (ưu tiên) / Colab, GPU T4 16GB** | calibration, benchmark 4K–16K, NIAH, LongBench, GSM8K |

Quy tắc mọi phiên GPU (bắt buộc từ Phase 5):
1. Notebook chỉ là **runner mỏng**: clone repo tại commit cố định, in `git rev-parse HEAD`, chạy `python experiments/phaseN_<tên>.py --nonce <N> 2>&1 | tee logs/<tag>.log`.
2. Không viết logic thí nghiệm trong cell notebook.
3. Cuối phiên: zip `logs/` + `results/`, tải về, chép vào `logs/` local để trích dẫn file:dòng.
4. Mỗi script GPU có `--smoke` chạy được trên CPU với model nhỏ; chạy smoke trước khi tốn quota.

## 1. Quyết định đã chốt

| ID | Quyết định | Hệ quả |
|---|---|---|
| D1 | V̂ chỉ tồn tại tạm thời khi tính attention, **không giữ trong cache** (Spec:100) | Phase 8.2 |
| D2 | Chỉ số bộ nhớ chính = **bộ nhớ KV cache lâu dài** (đếm mọi tensor cache giữ); Peak VRAM là chỉ số phụ | Spec §5.4 đã sửa; Phase 6, 8, 11.1 |
| D3 | Điểm H2O tính **đúng công thức gốc theo từng khối query (chunked)**; prefill chạy SDPA; dùng chung cho H2O và VECTOR | Phase 10; không lệch khỏi H2O gốc, không cần sửa Spec |
| D4 | Nếu R² held-out < 0.70: vẫn tiếp tục, báo cáo là phát hiện, thêm thí nghiệm p_a theo layer | Spec §2.3 đã sửa; Phase 9, 11.4 |

## 2. Bất biến bắt buộc (I1–I6)

Mọi thay đổi cache phải giữ các bất biến này; brief đụng tới cache phải chép lại các bất biến liên quan thành tiêu chí chấp nhận.
- **I1.** position_id của token mới == `seen_tokens` trước bước đó (vị trí tuyệt đối), kể cả sau khi tỉa.
- **I2.** kv_len của mask == độ dài K **sau** `update()` ở mọi layer.
- **I3.** Mọi layer có cùng độ dài K.
- **I4.** `accumulated_scores[layer].shape[-1] == K[layer].shape[-2]` sau `update()` và sau khi hook cộng điểm; tỉa phải gather điểm (và `_cached_eps`, positions) theo cùng selected indices.
- **I5.** K trong cache giữ RoPE của vị trí gốc, không re-rotate.
- **I6.** `compression_ratio = 0` cho output greedy giống hệt Full-FP16.

Bổ sung theo D1/D2:
- **I7.** Bộ nhớ thật mà cache giữ (mọi tensor, gồm `layers[i].keys/values`) == `get_memory_bytes()`.
- **I8.** Ở cùng `target_budget`, bộ nhớ KV lâu dài của VECTOR ≤ H2O.

Hợp đồng độ dài (chi tiết trong `CLAUDE.md`): `seen_tokens` chỉ tăng → position/RoPE; `physical_len` → shape và bộ nhớ; `mask_kv_len` == K sau update; `target_budget` điều khiển Budget Swap; `max_physical_budget` là hard cap, xung đột thì raise lỗi.

## 3. Lịch sử đã hoàn thành (Phase 0–4)

| Phase | Nội dung | Bằng chứng |
|---|---|---|
| 0 | Xác nhận nguồn mask/position trên HEAD | `logs/phase0_current_head.log` |
| 1 | Test tái hiện + chuẩn hóa contract mask (guard, không slice) | `logs/phase1_mask_green.log`, `logs/phase1_mask_mutation.log` |
| 2 | Logical/physical length cho H2O và VECTOR, I1–I5 | `logs/phase2_verify_i1_i5_fixed.log` |
| 2b | Tách `target_budget` / `max_physical_budget`, hard cap từ prefill | `logs/phase2b_memory_green2.log`, `logs/phase2b_memory_mutation.log`, `logs/phase2b_final_full_pytest.log` |
| 3 | I1–I6 trên Qwen2 nhỏ, 20 bước decode, NaN/memory bound | `logs/phase3_i1_i5_final.log`, `logs/phase3_i6_no_compression_fixed.log`, `logs/phase3_nan_memory_bound.log` |
| 4 (một phần) | Benchmark CPU nhỏ Full/H2O/VECTOR | `reports/phase4_benchmark_small.md` |

Việc còn dở từ lộ trình cũ đã chuyển vào các phase dưới đây:
- "Không giữ tensor lớn cũ sau khi thay cache" → Phase 8.
- NIAH single/multi needle → Phase 11.
- Kết luận về memory/throughput/recall → Phase 11.

⚠️ Số `persistent_kv_bytes` của VECTOR trong `reports/phase4_benchmark_small.*` thấp hơn thực tế (vi phạm I7, xem Review P0). Không dùng các số đó trong báo cáo đồ án.

## 4. Các phase tiếp theo

Trạng thái: `[ ]` chưa làm · `[~]` đang làm · `[x]` xong (chỉ đánh dấu sau khi review CHẤP NHẬN). Số phiên GPU là ước lượng.

### Phase 5 — Hạ tầng bằng chứng cho Kaggle/Colab · 💻 + 🟢×1
- [x] **5.1** `experiments/_header.py` dùng chung: nonce, timestamp, python/torch/transformers, device, seed, sha256 script, git commit. (Review: `reports/reviews/5.1_experiment_header_review.md`, CHẤP NHẬN 2026-10-24)
- [ ] **5.2** `notebooks/kaggle_runner.ipynb`: clone theo commit, chạy một script, zip logs.
- [ ] **5.3** 🟢 `phase5_gpu_smoke.py`: load Qwen2.5-1.5B, 1 bước decode Full; log tải về có đủ header và `[EXIT] code=0`.

### Phase 6 — Đo hiện trạng, chưa sửa · 💻 + 🟢×1
- [ ] **6.1** 💻 `phase6_memory_truth.py`: đếm bytes mọi tensor cache giữ, so với `get_memory_bytes` (I7, I8) cho Full/H2O/VECTOR.
- [ ] **6.2** 🟢 cùng script trên Qwen2.5-1.5B ở 4K: `memory_allocated` sau load / sau prefill / sau decode, `max_memory_allocated`.
- [ ] **6.3** 💻 test tái hiện: token Approximate được nâng lên Retain mang theo V̂.
- **Điểm dừng:** nộp số liệu nền cho người dùng trước khi sửa.

### Phase 7 — Refactor thuần, không đổi hành vi · 💻
- [ ] **7.1** `src/kv_core/base_cache.py` (`PrunedCacheBase`: seen/physical len, `get_mask_sizes`, `_sync_dynamic_layers`).
- [ ] **7.2** `ScoreAccumulator` + `split_sink_recent` dùng chung.
- [ ] **7.3** Gộp hai hook manager, giữ alias cũ; bỏ `except: pass`.
- [ ] **7.4** Xóa `slice_counter` chết, sửa docstring `mask_patch`.
- **Ra:** test cũ xanh trước/sau; `vector_cache.py`, `h2o_cache.py` ≤ 300 dòng; output 6.1 không đổi.

### Phase 8 — Bộ nhớ VECTOR đúng Spec (D1) · 💻 → 🟢×1
- [ ] **8.1** `get_memory_bytes` đếm mọi tensor cache giữ, tách `kv_bytes` / `metadata_bytes` (I7).
- [ ] **8.2** Không giữ V̂ sau `update()`; trước khi sửa phải grep mọi chỗ HF đọc `layers[i].values`. Ba log: FAIL / PASS / đột biến FAIL. Chuỗi token greedy không đổi.
- [ ] **8.3** Nếu 6.3 FAIL: Retain chỉ nhận V thật.
- [ ] **8.4** 🟢 chạy lại 6.2: I7, I8 pass trên Qwen2.5-1.5B.

### Phase 9 — Calibration lại (D4) · 🟢×1–2 · song song với Phase 7–8
- [ ] **9.1** Nạp Wikitext-103/C4 (32–64 mẫu, Spec:96), tách calib/held-out, R² held-out theo layer/head.
- [ ] **9.2** So sánh R² có/không khử RoPE.
- **Điểm dừng:** nếu R² < 0.70, báo cáo và áp dụng D4 trước khi sang Phase 11.

### Phase 10 — Chấm điểm H2O chunked, prefill bằng SDPA (D3) · 💻 → 🟢×1
- [ ] **10.1** Hook lấy q đã áp RoPE ở mỗi layer khi chạy SDPA; điểm = Σ_i softmax(q_i·Kᵀ) tính theo khối c query (bộ nhớ c×L).
- [ ] **10.2** Dùng chung cho H2O và VECTOR qua `kv_core`; decode 1 query tính trực tiếp.
- [ ] **10.3** 💻 so với `attn_weights` eager trên Qwen2 nhỏ: điểm khớp trong ngưỡng (in expected/actual); đột biến (bỏ RoPE khỏi q) phải FAIL.
- [ ] **10.4** 🟢 ở 4K: chuỗi greedy giống bản eager; prefill 16K chạy được với cả ba cache trên T4.

### Phase 11 — Benchmark chính (Spec §5) · 🟢×3–4
- [ ] **11.1** Bộ nhớ + tốc độ: p_c ∈ {0, 50, 75, 85}%, context {4K, 8K, 16K}, decode 128 token, cùng attention implementation.
  - **Điểm dừng:** nếu VECTOR không đạt I8, dừng, không chạy tiếp để "tìm số đẹp".
- [ ] **11.2** Single-needle NIAH heatmap (context × depth).
- [ ] **11.3** Multi-needle 3 và 5 kim.
- [ ] **11.4** Quét p_a ∈ {0.1, 0.2, 0.3, 0.4} ở p_c = 75%, và p_a theo layer nếu áp dụng D4.

### Phase 12 — Chất lượng mở rộng · 🟢×2–3
- [ ] **12.1** LongBench NarrativeQA + Qasper (tập con cố định), F1/ROUGE-L.
- [ ] **12.2** GSM8K 200 câu, Exact Match.

### Phase 13 — Mở rộng của Spec (tùy chọn) · 💻 + 🟢
- [ ] **13.1** Baseline INT8/INT4 (kiểm tra `QuantizedCache` của transformers 5.20.1 trước khi tự viết).
- [ ] **13.2** Llama-3.2-1B: dùng `rotary_emb` của model (xử lý `rope_scaling`), calibration riêng.

### Phase 14 — Đóng gói · 💻
- [ ] **14.1** README chạy lại kết quả bằng một lệnh.
- [ ] **14.2** Báo cáo đồ án, slide (Spec §6 tuần 13–15).

## 5. Thứ tự và đường găng

```
P5 ─► P6 ─► P7 ─► P8 ─► P10 ─► P11 ─► P12 ─► P14
                    ▲
P9 (GPU, song song) ┘                  P13 (tùy chọn)
```
Đường găng: P5 → P6 → P7 → P8 → P10 → P11. P9 chỉ cần GPU, không đụng code cache, nên làm trong lúc chờ P7–P8.
