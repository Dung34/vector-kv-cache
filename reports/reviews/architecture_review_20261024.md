# Nhận xét kiến trúc dự án — 2026-10-24

Phạm vi: `src/` (baselines, vector_cache, utils), cách đo bộ nhớ trong `experiments/`. Không sửa code.

## Log đã sinh trong phiên này
| Log | sha256 | Nội dung |
|---|---|---|
| `logs/review_arch_env.log` | `66a23447…9662b` | transformers 5.20.1, torch 2.8.0+cpu, `DynamicCache` có `layers` |
| `logs/review_arch_pytest.log` | `5a75e093…f38f7` | `pytest tests/ -q` → 43 passed |
| `logs/review_arch_static.log` | `3e86a721…6745` | wc -l, grep, diff hai hook, danh sách method trùng |
| `logs/review_arch_probe_layers.log` | `499cf9ca…62b` | probe CPU: bộ nhớ thật của VECTOR vs H2O |

Probe: script tạm ngoài repo `C:\Users\Dung\AppData\Local\Temp\claude\d--MyProject-ProjectIII\74563b8f-e2f9-4af8-8f3a-9eaa6a342a8e\scratchpad\probe_layers_mem.py` (B=1, H=2, D=16, prefill 64, target_budget=16, 3 bước decode, chỉ layer 0). Ghi chú: `[EXIT] code=0` lấy từ `$?` sau `tee`, không phải exit code của python; output không có traceback.

## 1. Điểm tốt
| Khẳng định | Nhãn | Bằng chứng |
|---|---|---|
| Test suite hiện tại xanh | [QUAN SÁT] | `logs/review_arch_pytest.log`: "43 passed in 22.50s" |
| VECTOR tách đúng tầng §5: partitioner (policy thuần tensor), reconstructor (math), cache (container), hook | [QUAN SÁT] | `src/vector_cache/partitioner.py:20-202`, `reconstructor.py:11-153`, `vector_cache.py:25` |
| Mọi package export class public ở `__init__.py` | [QUAN SÁT] | `src/vector_cache/__init__.py`, `src/baselines/__init__.py`, `src/calibration/__init__.py`, `src/evaluation/__init__.py` |
| Hợp đồng logical/physical (seen_tokens chỉ tăng, physical_len theo `_key_cache`) cài nhất quán ở cả hai cache | [QUAN SÁT] | `h2o_cache.py:86-95`, `vector_cache.py:125-134` |

## 2. Vấn đề — xếp theo mức ảnh hưởng tới kết luận nghiên cứu

### P0. VECTOR giữ V̂ đầy đủ trong `self.layers`, nhưng `get_memory_bytes` không đếm
| Khẳng định | Nhãn | Bằng chứng |
|---|---|---|
| Sau mỗi bước decode, `layers[0].values` của VECTOR dài bằng K (24), còn V thật lưu chỉ 8 | [QUAN SÁT] | `logs/review_arch_probe_layers.log:2-4` |
| Bộ nhớ thật của VECTOR = 7168 B, `get_memory_bytes` báo 4096 B; H2O = 4096 B, `layers.values` trùng object | [QUAN SÁT] | `logs/review_arch_probe_layers.log:2-7` |
| Nguyên nhân: `_partition_and_prune` dựng `v_returned` (N_K × d) rồi gán vào `layers[i].values` | [QUAN SÁT] | `vector_cache.py:302-303`, `vector_cache.py:456-457` |
| Các thí nghiệm bộ nhớ (phase2b, phase4, phase_review) dùng `get_memory_bytes` | [QUAN SÁT] | `logs/review_arch_static.log:19-22` |
| ⇒ Số "persistent_kv_bytes" của VECTOR trong báo cáo phase4 đang thấp hơn thực tế, và ở cấu hình probe VECTOR tốn 1.75× H2O chứ không bằng | [SUY LUẬN] | từ hai dòng QUAN SÁT trên (7168/4096) |
| Tỷ lệ trên ở model thật Qwen2.5-1.5B | [CHƯA KIỂM CHỨNG] | chạy `scripts/run_memory_benchmark.py` có đo `torch.cuda.memory_allocated` sau decode, so với `get_memory_bytes` |

Việc này cần người dùng chọn hướng (quyết định nghiên cứu): (a) không giữ V̂ trong `layers` (chỉ trả về cho attention rồi bỏ), hay (b) giữ nhưng đếm vào bộ nhớ và báo cáo trung thực.

### P1. Token Approximate có thể được "thăng hạng" lên Retain với V̂ coi như V thật
| Khẳng định | Nhãn | Bằng chứng |
|---|---|---|
| Mỗi bước decode, `v_full_prev` được dựng lại bằng V̂ ở vị trí Approximate rồi đưa cả vào partition | [QUAN SÁT] | `vector_cache.py:424-431` |
| `v_retain` được gather từ `v_full` đó, không phân biệt V thật hay V̂ | [QUAN SÁT] | `partitioner.py:191-192` |
| ⇒ Nếu một token A ở bước t lọt vào Retain ở bước t+1, V̂ của nó sẽ bị lưu như V thật; sai số tích lũy mà `cached_eps` không phản ánh | [SUY LUẬN] | từ hai dòng trên |
| Trường hợp này có xảy ra thật không | [CHƯA KIỂM CHỨNG] | test CPU: theo dõi `cand_indices` qua 2 bước, kiểm tra giao giữa approx(t) và retain(t+1) |

### P1. Trùng lặp code H2O ↔ VECTOR và vượt giới hạn 300 dòng
| Khẳng định | Nhãn | Bằng chứng |
|---|---|---|
| `vector_cache.py` 468 dòng, `h2o_cache.py` 393 dòng (vượt §5) | [QUAN SÁT] | `logs/review_arch_static.log:2-3` |
| 6 method trùng tên ở cả hai cache: `get_mask_sizes`, `register_attention_scores`, `_sync_dynamic_layers`, `get_memory_bytes`, `_physical_len`, `get_seq_length` | [QUAN SÁT] | `logs/review_arch_static.log:26-37` |
| Logic chia sink/recent bị chép hai nơi | [QUAN SÁT] | `h2o_cache.py:305-310` vs `partitioner.py:55-64` |
| Hai hook manager gần như giống hệt (diff sau khi chuẩn hóa tên chỉ 10 dòng, toàn comment/docstring) | [QUAN SÁT] | `logs/review_arch_static.log:24` |
| ⇒ Sửa bất biến ở một cache rất dễ quên cache kia; tách base class sẽ đưa cả hai file về dưới 300 dòng | [SUY LUẬN] | từ các dòng trên |

### P2. Lỗi bị nuốt im lặng và code chết
| Khẳng định | Nhãn | Bằng chứng |
|---|---|---|
| `register_hooks` bắt `Exception` rồi `pass` khi patch mask và khi đăng ký pre-hook decode | [QUAN SÁT] | `vector_hook.py:61-65`, `vector_hook.py:78-82` |
| `slice_counter` không bao giờ được tăng, nhưng `phase0b_deep.py` đọc nó như một số đo | [QUAN SÁT] | `logs/review_arch_static.log:8-16` |
| Docstring mask_patch nói "sliced", code thực tế chỉ assert | [QUAN SÁT] | `mask_patch.py:18-21` vs `mask_patch.py:40-44` |
| Fallback `class DynamicCache: pass` khi import lỗi che mất lỗi môi trường | [QUAN SÁT] | `vector_cache.py:14-19`, `h2o_cache.py:9-14` |
| Số head mặc định 12/2 hard-code khi thiếu config | [QUAN SÁT] | `vector_hook.py:33-37` |

### P2. Giới hạn thiết kế chưa ghi thành văn bản
| Khẳng định | Nhãn | Bằng chứng |
|---|---|---|
| Khi decode 1 token, pre-hook thay mask bằng `{"full_attention": None}` nên assert của mask_patch không bao giờ kiểm ở decode | [SUY LUẬN] | `vector_hook.py:72-75` + `mask_patch.py:40` (chỉ assert khi mask khác None) |
| ⇒ Không có padding mask khi decode: batch > 1 có padding sẽ sai | [SUY LUẬN] | cùng tiền đề |
| Mỗi bước decode, mỗi layer: tái tạo toàn bộ V̂ của N_A token và tạo ma trận so sánh N_K × N_R | [QUAN SÁT] | `vector_cache.py:425`, `partitioner.py:175-177` |
| Chi phí này đáng kể so với decode ở budget lớn | [CHƯA KIỂM CHỨNG] | profile `update()` với target_budget ∈ {256, 1024, 2048} |

## 3. Kiến trúc đề xuất
```
src/kv_core/                     # mới, dùng chung
  base_cache.py   PrunedCacheBase(DynamicCache): seen/physical len, get_mask_sizes,
                  _sync_dynamic_layers, get_memory_bytes (đếm cả layers.values nếu khác object,
                  positions, scores, eps — tách kv_bytes / metadata_bytes)
  scores.py       ScoreAccumulator (register_attention_scores + pad/gather)
  budget.py       split_sink_recent(total_len, target_budget, ...) — dùng chung H2O & VECTOR
  hooks.py        AttentionHookManager (một class, nhận mọi cache có register_attention_scores)
src/baselines/h2o_cache.py       chỉ còn chính sách H2O (< 200 dòng)
src/vector_cache/vector_cache.py chỉ còn điều phối partitioner/reconstructor (< 250 dòng)
```
`H2OAttentionHookManager` / `VectorAttentionHookManager` giữ lại làm alias để không vỡ notebook.

## 4. Thứ tự brief đề xuất (mỗi brief một hành vi)
1. **Đo trước, chưa sửa:** experiment `phase5_memory_truth.py` đo bộ nhớ thật (CPU: tổng bytes mọi tensor cache giữ; GPU: `memory_allocated`) cho Full/H2O/VECTOR. → Chờ người dùng chọn (a) hay (b) ở P0.
2. Test tái hiện P1 (thăng hạng Approximate → Retain). Nếu FAIL mới viết brief sửa.
3. Trích `PrunedCacheBase` + `ScoreAccumulator` (refactor thuần, test cũ phải xanh trước và sau).
4. Trích `split_sink_recent` dùng chung.
5. Gộp hai hook manager; bỏ `except: pass`, thay bằng lỗi rõ ràng.
6. Xóa `slice_counter` hoặc cài đúng; sửa docstring mask_patch.

## 5. Chưa làm / chưa kiểm chứng
- Chưa đọc `src/calibration/`, `src/evaluation/`, `scripts/`, `notebooks/` chi tiết.
- Chưa chạy model thật; mọi kết luận bộ nhớ chỉ ở mức probe CPU 1 layer.
- P1 (thăng hạng) mới là suy luận từ code.

## 7. Đối chiếu với Spec (`docs/PROJECT_SPECIFICATION.md`)
Log thêm: `logs/review_spec_grep.log` (sha256 `ce3dabca…411e`).

| Khẳng định | Nhãn | Bằng chứng |
|---|---|---|
| Spec yêu cầu chỉ cấp VRAM cho K của Retain+Approx và V của Retain ⇒ P0 phải đi hướng (a), hướng (b) trái Spec | [SUY LUẬN] | Spec:100, Spec:42 ("không làm tăng VRAM") + P0 ở trên |
| Spec đặt việc tái tạo V̂ trong hook "ngay trước phép nhân" lúc decode; code tái tạo trong `update()` và lưu vào `layers` | [QUAN SÁT] | Spec:101 vs `vector_cache.py:302-303` |
| Chỉ số bộ nhớ chính của Spec là Peak VRAM (`max_memory_allocated`), không phải `get_memory_bytes` | [QUAN SÁT] | Spec:129, Spec:143; chỉ `memory_tracker.py:33` dùng (`logs/review_spec_grep.log:14`) |
| Kích thước pool ứng viên khớp Spec: n_k = (1 − p_c + p_a)·L | [SUY LUẬN] | đại số từ `partitioner.py:67-71` với target = (1−p_c)·L; Spec:71 |
| Baseline Uniform Quantization INT4/INT8 chưa có | [QUAN SÁT] | Spec:114; grep chỉ thấy cast `torch.int8` trong topk (`logs/review_spec_grep.log:2-6`) |
| LongBench và GSM8K chưa có code đánh giá | [QUAN SÁT] | Spec:133-134; `logs/review_spec_grep.log:7-8` rỗng |
| Spec có Llama-3.2-1B, nhưng reconstructor tự tính RoPE thuần `1/θ^(2i/d)`, không xử lý `rope_scaling` (llama3); `rotary_emb` được nhận nhưng không dùng | [QUAN SÁT] | Spec:11; `reconstructor.py:66`; `logs/review_spec_grep.log:9-12` |
| ⇒ Khử RoPE cho Llama-3.2 sẽ sai, W_OLS/ε của Llama không đáng tin | [SUY LUẬN] | từ dòng trên |
| Spec ghi kế thừa `Cache`, code kế thừa `DynamicCache` — lệch nhỏ, chấp nhận được | [QUAN SÁT] | Spec:98; `vector_cache.py:25` |

## 6. Bất ngờ
- VECTOR, với lý thuyết "N_K + N_R = 2·budget" bằng H2O, trong probe lại giữ nhiều bộ nhớ hơn H2O (`logs/review_arch_probe_layers.log:2` vs `:5`). Cái lợi cốt lõi về bộ nhớ của phương pháp cần đo lại trước khi viết thêm kết quả.
