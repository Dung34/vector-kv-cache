# Lộ trình: tái hiện VECTOR trên KVPress (Qwen3-0.6B)

Lộ trình và danh sách task duy nhất. Nguồn chuẩn: `docs/PROJECT_SPECIFICATION.md`. Quy trình làm việc: `AGENTS.md` §0 và `CLAUDE.md`.
Mỗi milestone (R0, R1, …) = **một brief** (`tasks/briefs/<R?>_<tên>.md`), **một báo cáo**, **một lần review**.

## Quyết định
| ID | Nội dung |
|---|---|
| D1 | Framework: NVIDIA KVPress, commit `91773cb8b5713fe6ad54e3e7a97dd343fb40d37b` |
| D2 | Mô hình duy nhất: `Qwen/Qwen3-0.6B` |
| D3 | Base scorer: KeyDiff + SnapKV trước; KVzip / PyramidKV để mở rộng |
| D4 | $p_a$ theo paper Eq. 1 |
| D5 | Bộ nhớ: chỉ số chính là bộ nhớ KV lâu dài, Peak VRAM là phụ |
| D6 | $R^2$ thấp hơn paper thì vẫn tiếp tục và báo cáo |
| D7 | Chạy chính trên **RTX 3050 cục bộ**; dùng Kaggle T4 chỉ khi cần > 4 GB (ngữ cảnh dài). *Chờ người dùng đồng ý cài môi trường GPU cục bộ (R1.1).* |

**Không dùng làm kết quả:** mọi số liệu trong `reports/` có từ trước ngày 2026-10-25, vì có thể đã chạy fp16 + eager trên T4 và bị NaN. Code cũ trong `src/` (H2O tự viết, cache VECTOR nén theo từng bước decode) là legacy; riêng `src/calibration/` có thể dùng lại cho R3.

## Milestones
Trạng thái: `[ ]` chưa làm · `[~]` đang làm · `[x]` xong.

### R0. Khảo sát KVPress (chỉ đọc)
- [ ] Ghi chú kỹ thuật `docs/kvpress_notes.md`, đọc ở đúng commit: `BasePress.forward_hook` / `compress`, `ScorerPress`, `KeyDiffPress`, `SnapKVPress` (+ `query_aware`), `pipeline.py`, benchmark `longbench` và `needle_in_haystack`, `evaluation/evaluate.py`.
- [ ] Trả lời 4 câu hỏi: VECTOR gắn vào đâu; lấy $K$ trước RoPE thế nào; benchmark được chạy và chấm điểm thế nào; phiên bản transformers / torch tương thích.

### R1. Môi trường + baseline chạy được
- [ ] **R1.1** Môi trường GPU cục bộ: venv mới `.venv_gpu` (torch CUDA + KVPress đúng commit). Không đụng `.venv_v5`.
- [ ] **R1.2** Chạy No-compression, KeyDiff, SnapKV ($p_c = 0.75$) trên 1–2 nhiệm vụ LongBench nhỏ và vài mẫu NIAH; kiểm tra không NaN; ghi bộ nhớ tối đa và thời gian chạy.

### R2. Baseline so với paper
- [ ] Chọn tập con LongBench chạy vừa phần cứng (ghi lý do). Chạy No-compression, KeyDiff, SnapKV × 4 mức $p_c$. So với paper Table 3 từng ô.
- **Điểm dừng:** baseline lệch xa paper thì dừng, tìm nguyên nhân trước khi viết VECTOR.

### R3. Calibration $W_{OLS}$
- [ ] Thu $K$ trước RoPE và $V$ trên C4 (chuỗi 4,096); fit OLS theo layer, thử cả theo từng head lẫn gộp head; đo $R^2$ held-out; so với 0.9392.

### R4. VECTOR press
- [ ] Press bọc ngoài `ScorerPress`: Relax → ε → Truncate (chế độ chất lượng).
- [ ] Kiểm tra: số key/value đúng công thức; $p_a = 0$ trùng press gốc; $W_{OLS}$ "hoàn hảo" (dùng $V$ thật) trùng kết quả giữ toàn bộ tập ứng viên.

### R5. Tái hiện
- [ ] LongBench: KeyDiff ± VECTOR, SnapKV ± VECTOR × 4 mức $p_c$; so với Table 3.
- [ ] Độ nhạy $p_a$ ở $p_c = 0.75$ (Figure 3, thu nhỏ).
- [ ] NIAH ở $p_c = 0.90$ (Figure 4, thu nhỏ).

### R6. Chốt số liệu cho báo cáo (chế độ bằng chứng)
- [ ] Chạy lại các số đưa vào báo cáo theo AGENTS.md §6: commit sạch, nonce, log.

### R7. Mở rộng
- [ ] KVzip, PyramidKV ± VECTOR; đo bộ nhớ KV thật; (tùy chọn) chế độ bỏ $V$ vật lý.

### R8. Báo cáo
- [ ] Báo cáo đồ án, slide, README tái lập.

Thứ tự: R0 → R1 → R2 → R4 → R5 → R6 → R7 → R8. R3 làm song song với R2.
