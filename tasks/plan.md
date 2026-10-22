# Roadmap: Ổn định H2O/VECTOR KV Cache

## Mục tiêu

Đưa H2O và VECTOR từ trạng thái prototype có test đơn vị sang trạng thái có thể chạy decode ổn định trên Qwen2, giữ đúng vị trí tuyệt đối của token, không lệch attention mask, không vượt giới hạn bộ nhớ đã cấu hình, và có benchmark đáng tin cậy.

Phạm vi ưu tiên hiện tại là correctness và memory safety. Benchmark chất lượng, throughput và NIAH chỉ được thực hiện sau khi các invariant lõi đã pass.

## Trạng thái hiện tại

- [x] Đã ghi nhận và lưu các experiment/report Phase 0b và Phase 2.
- [x] H2O đã có phân biệt sơ bộ giữa `seen_tokens` và physical cache length.
- [ ] VECTOR chưa được kiểm chứng đầy đủ theo cùng contract logical/physical length.
- [ ] Test suite hiện chưa xanh: test mask patch đang fail.
- [ ] `max_budget` chưa bảo vệ peak memory của prefill đầu tiên.
- [ ] Chưa có bằng chứng Phase 3 đầy đủ cho cả H2O và VECTOR trên model Qwen2 nhỏ.
- [ ] Chưa được phép chạy benchmark 4096 token như tiêu chí nghiệm thu cuối.

Bằng chứng hiện tại:

- `logs/review_20261022_pytest.log`: 28 test pass, 1 test fail.
- `logs/phase_review_memory.log`: prefill 64 token vẫn lưu physical length 64 với `max_budget=16` ở cả H2O và VECTOR.
- `tasks/fixh2ovector.md`: đặc tả invariant I1–I6 và thứ tự điều tra/sửa.

## Hợp đồng dữ liệu bắt buộc

Mọi cache phải phân biệt rõ ba đại lượng:

1. `seen_tokens`: tổng số token tuyệt đối đã xử lý; chỉ tăng, không giảm.
2. `physical_len`: số token K/V thực sự đang được lưu sau prune.
3. `mask_kv_len`: chiều dài KV mà causal attention mask sử dụng.

Quy tắc sử dụng:

- `seen_tokens` dùng cho `cache_position`, `position_ids` và RoPE.
- `physical_len` dùng cho shape K/V thật và memory accounting.
- `mask_kv_len` phải khớp K sau `cache.update()` ở mọi layer.
- `accumulated_scores`, `_cached_eps` và position metadata phải được prune/gather cùng selected indices.
- `target_budget` điều khiển phân bổ Budget Swap; không được dùng nó làm hard cap physical nếu thuật toán cần `N_K > target_budget`.
- `max_physical_budget` là hard cap tùy chọn cho physical K/V và peak-memory safety.
- Nếu `target_budget` và `max_physical_budget` xung đột, phải raise lỗi cấu hình rõ ràng; không tự động cắt key/value để che xung đột.

## Phase 0 — Xác nhận baseline trên HEAD hiện tại — COMPLETED

**Mục tiêu:** xác nhận behavior của đúng phiên bản code đang chuẩn bị sửa.

**Công việc:**

- [x] Chạy inspect source của `Qwen2Model.forward`, `DynamicCache.get_seq_length` và `get_mask_sizes`.
- [x] Ghi phiên bản Python, PyTorch, Transformers, device và seed.
- [x] Chạy Qwen2 config nhỏ: 2 layer, hidden size 64, 4 attention heads, 2 KV heads, context 64.
- [x] Ghi theo từng layer: `seen_tokens`, `physical_len`, `mask_kv_len`, `position_id`, `scores_len`.
- [x] Xác định chính xác mask được tạo trước hay sau `cache.update()`.

**Acceptance criteria:**

- [x] Có một log Phase 0 mới, chạy bằng script có nonce.
- [x] Trả lời được mask length lấy từ API nào.
- [x] Trả lời được position ID lấy từ API nào.
- [x] Không thay đổi production code trong phase này.

**Checkpoint:** đã đạt; log: `logs/phase0_current_head.log`.

## Phase 1 — Tạo test tái hiện và sửa contract mask — COMPLETED

**Mục tiêu:** biến lỗi shape thành test regression phản ánh đúng invariant, không dùng workaround để che lỗi.

**Công việc:**

- [x] Tạo test tái hiện mismatch giữa mask và K sau prune.
- [x] Chạy test trên code hiện tại và lưu log FAIL.
- [x] Quyết định `mask_patch` chỉ là guard/diagnostic; logic chiều dài phải nằm ở cache/model contract.
- [x] Sửa test `mask_patch` để phân biệt case hợp lệ và mismatch thật.
- [x] Kiểm tra mask lệch và mask đã căn chỉnh.

**Acceptance criteria:**

- [x] Code/test contract cũ FAIL đúng invariant.
- [x] Code/test contract mới PASS.
- [x] Mutation làm mask length sai phải FAIL.
- [x] Guard không tự slice/pad mask để che mismatch.

**Dependencies:** Phase 0.

**Evidence:** `logs/phase1_mask_green.log`, `logs/phase1_mask_mutation.log`, `logs/phase1_full_pytest.log`.

## Phase 2 — Hoàn thiện logical/physical length cho H2O và VECTOR — COMPLETED

**Mục tiêu:** áp dụng cùng một contract cho hai cache.

**Công việc:**

- [x] Hoàn thiện `seen_tokens` trong H2O.
- [x] Implement và kiểm thử cùng contract trong VECTOR.
- [x] Bảo đảm `get_seq_length()` trả logical length.
- [x] Bảo đảm `get_mask_sizes()` dùng physical length và query length thực tế.
- [x] Kiểm tra mọi layer có cùng physical K length.
- [x] Kiểm tra RoPE không bị re-rotate sau prune.

**Acceptance criteria:**

- [x] I1: position ID token mới bằng `seen_tokens` trước bước decode.
- [x] I2: mask KV length bằng K length sau update.
- [x] I3: mọi layer có cùng K length.
- [x] I4: scores length bằng K length trước/sau hook và prune.
- [x] I5: K giữ RoPE theo vị trí gốc.

**Dependencies:** Phase 1.

**Evidence:** `logs/phase2_verify_i1_i5_fixed.log`, `logs/phase2_final_full_pytest.log`.

## Phase 2b — Bảo vệ memory budget và peak memory

**Mục tiêu:** tách ngân sách thuật toán khỏi giới hạn physical memory, đồng thời loại bỏ tăng trưởng bộ nhớ ngoài dự kiến.

**Công việc:**

- [x] Đổi API/cache state để phân biệt `target_budget` và `max_physical_budget`.
- [x] Giữ Budget Swap theo `target_budget`; không dùng `target_budget` như physical hard cap.
- [x] Kiểm tra compatibility trước khi chạy; cấu hình xung đột phải raise lỗi rõ ràng.
- [x] Áp dụng `max_physical_budget` ngay ở prefill đầu tiên.
- [x] Kiểm tra prefill một chunk và nhiều chunk với `incoming_len > 1`.
- [x] Đo persistent KV bytes và auxiliary bytes riêng biệt.
- [ ] Đo peak temporary memory trong `torch.cat`, partition và reconstruction.
- [ ] Prune đồng bộ K, V, positions, scores và reconstruction errors.
- [ ] Kiểm tra không giữ các tensor lớn cũ sau khi thay cache.

**Acceptance criteria:**

- [x] Với mọi bước update, `physical_len <= max_physical_budget` khi cap được cấu hình.
- [x] Cấu hình hợp lệ `target_budget=32, max_physical_budget=64` giữ được Budget Swap.
- [x] Cấu hình xung đột `target_budget=32, max_physical_budget=32` bị từ chối rõ ràng.
- [x] H2O và VECTOR đều bounded sau prefill và decode.
- [ ] Không cần dùng `empty_cache()` để làm cho invariant có vẻ đúng.
- [x] Có log chứa memory table được tính từ tensor thật.

**Dependencies:** Phase 2.

## Phase 3 — Nghiệm thu model Qwen2 nhỏ

**Mục tiêu:** xác nhận toàn bộ invariant trong flow model thật nhưng nhỏ, chạy CPU.

**Công việc:**

- [ ] Chạy H2O ít nhất 20 bước decode.
- [ ] Chạy VECTOR ít nhất 20 bước decode.
- [ ] Kiểm tra I1–I5 ở mọi bước.
- [ ] Kiểm tra I6: `compression_ratio=0` cho output greedy giống Full-FP16.
- [ ] Ghi bảng `step | seen_tokens | physical_len | mask_kv_len | position_id | scores_len`.
- [ ] Kiểm tra NaN, exception và memory bound.

**Acceptance criteria:**

- [ ] Tất cả I1–I6 pass cho H2O.
- [ ] Tất cả I1–I6 pass cho VECTOR.
- [ ] Không có mismatch mask/K.
- [ ] Không có output bất thường so với Full-FP16 ở chế độ không nén.

**Dependencies:** Phase 2b.

**Checkpoint:** nếu Phase 3 chưa pass, không chạy benchmark 4096 token.

## Phase 4 — Benchmark thật và đánh giá chất lượng

**Mục tiêu:** đo đúng trade-off memory, latency và chất lượng sau khi correctness đã ổn định.

**Công việc:**

- [ ] Benchmark Full FP16, H2O và VECTOR.
- [ ] Đo riêng prefill latency, decode throughput và inter-token latency.
- [ ] Đo peak allocated/reserved VRAM và persistent KV bytes.
- [ ] Chạy single-needle và multi-needle NIAH.
- [ ] So sánh output token đầu với Full-FP16.
- [ ] Lưu JSON, bảng Markdown và log đầy đủ nonce/exit code/hash.

**Acceptance criteria:**

- [ ] Báo cáo có context length, compression ratio, device và seed.
- [ ] Không dùng số liệu từ run chưa có log.
- [ ] Báo cáo tách rõ kết quả đo được và suy luận.
- [ ] Có kết luận riêng cho memory, throughput và recall.

**Dependencies:** Phase 3.

## Quy tắc commit

Mỗi phase chỉ commit sau khi verification tương ứng hoàn tất:

1. `test: add current-head cache diagnostics`
2. `test: add logical-physical cache invariants`
3. `fix: unify seen and physical cache lengths`
4. `fix: enforce hard memory budget during prefill`
5. `fix: align pruned attention mask contract`
6. `test: validate H2O and VECTOR on small Qwen2`
7. `benchmark: compare bounded-memory cache variants`

## Rủi ro và cách giảm thiểu

| Rủi ro | Mức độ | Cách xử lý |
|---|---:|---|
| Mask được tính trước `cache.update()` | Cao | Xác minh bằng hook/source inspection ở Phase 0 |
| `seen_tokens` bị thay bằng physical length | Cao | I1 và mutation test |
| Prefill vượt physical memory cap | Cao | Hard-cap test ngay trong Phase 2b |
| `target_budget` và `max_physical_budget` bị trộn nghĩa | Cao | Kiểm tra compatibility và raise lỗi cấu hình |
| Attention weights `[B,H,L,L]` gây OOM | Cao | Đo peak và giới hạn capture ở Phase 2b/4 |
| H2O pass nhưng VECTOR lệch contract | Cao | Chạy cùng bảng invariant cho cả hai |
| Benchmark đẹp nhưng không tái lập được | Trung bình | Bắt buộc nonce, seed, hash và log nguyên văn |

## Trạng thái dừng hiện tại

**Đang ở Phase 2b — bảo vệ memory budget và peak memory.**

Phase 0 đã xác nhận decode hiện tại có `position_id=64` và mask length 32 khớp physical K length 32 ở cả hai layer. Phase 1 đã chuẩn hóa test contract mask. Phase 2 đã pass I1–I5 cho H2O và VECTOR trong 20 bước decode. Phase 0 cũng xác nhận prefill vẫn lưu 64 token dù `active_budget=32`; đây là blocker memory cần xử lý ở Phase 2b.
