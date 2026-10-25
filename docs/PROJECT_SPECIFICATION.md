# BẢN ĐẶC TẢ ĐỀ TÀI ĐỒ ÁN PROJECT III (PROJECT SPECIFICATION)

> Phiên bản 2.0, 2026-10-25. Các phiên bản trước nằm trong lịch sử git.

## 1. THÔNG TIN CHUNG

* **Tên đề tài (Tiếng Việt):** Nghiên cứu và Hiện thực Phương pháp Nén KV Cache Bất đối xứng Dựa trên Tính Cộng tuyến K–V cho Mô hình Ngôn ngữ Nhỏ (SLMs).
* **Tên đề tài (Tiếng Anh):** Research and Implementation of Asymmetric KV Cache Compression via K–V Collinearity for Small Language Models.
* **Mã học phần:** Project III (Đồ án chuyên ngành).
* **Bài báo gốc:** Lin et al., *A Simple Plug-in for Improving Eviction-Based KV Cache Compression* (VECTOR), arXiv:2605.23258, 2026 (`docs/papers/`).
* **Mô hình:** `Qwen/Qwen3-0.6B`. Paper có đủ kết quả cho mô hình này: LongBench ở Table 3 (phụ lục A), $R^2_{global} = 0.9392$ ở Table 1.
  * Cấu hình đo được: 28 layer, 16 q-heads, 8 kv-heads, head_dim 128, không có attention bias, trọng số bf16 1,434 MB, KV cache 16-bit ≈ 112 KB/token (`logs/qwen3_06b_sizing.log`).
* **Phần cứng:** GPU cục bộ RTX 3050 Laptop 4 GB (Ampere, hỗ trợ bf16); Kaggle / Colab T4 16 GB cho ngữ cảnh dài.

## 2. BỐI CẢNH

### 2.1. Nút thắt bộ nhớ KV Cache
$$\text{Memory}_{\text{KVCache}} = 2 \times b \times N_{\text{layer}} \times N_{\text{kv\_heads}} \times d_{\text{head}} \times L$$
Với Qwen3-0.6B, KV cache ở 32K token (≈ 3.5 GB) lớn hơn hai lần trọng số mô hình.

### 2.2. Hạn chế của eviction nhị phân
SnapKV, KeyDiff, KVzip, PyramidKV… giữ hoặc xóa cả $K$ lẫn $V$ của mỗi token. Ở mức nén cao (75–90%), thông tin bị xóa không khôi phục được.

### 2.3. Cơ sở của VECTOR
1. **Bất đối xứng độ nhạy:** sai số ở $K$ bị khuếch đại qua softmax, sai số ở $V$ chỉ lan truyền tuyến tính. Vì vậy giữ $K$ chính xác, chỉ xấp xỉ $V$ (paper phụ lục E).
2. **Cộng tuyến K–V:** sau khi khử RoPE ($R_m^{-1}K$), $V \approx W_{\text{OLS}}K$ với $W_{\text{OLS}}$ hiệu chuẩn offline.
3. **VECTOR là plug-in:** gắn vào base scorer có sẵn, thêm trục "khả năng tái tạo" bên cạnh "tầm quan trọng".

## 3. MỤC TIÊU

1. **Tái hiện (ưu tiên số 1):** hiện thực VECTOR thành một press của KVPress, tái hiện kết quả của paper trên `Qwen3-0.6B` với **KeyDiff** và **SnapKV**, ở các mức $p_c \in \{0.25, 0.50, 0.75, 0.90\}$.
   * **Tiêu chí:** (a) baseline (không nén, KeyDiff, SnapKV) gần với paper Table 3 trên cùng tập nhiệm vụ, sai lệch ghi rõ từng ô; (b) **hướng** cải thiện của "+VECTOR" so với baseline khớp paper ở $p_c \in \{0.75, 0.90\}$; (c) $R^2$ held-out gần 0.9392.
2. **Mở rộng (sau khi tái hiện đạt):** thêm base scorer KVzip, PyramidKV; đo bộ nhớ KV thật; (tùy chọn) chế độ bỏ $V$ vật lý.

## 4. PHƯƠNG PHÁP

### 4.1. Framework
* **NVIDIA KVPress**, commit `91773cb8b5713fe6ad54e3e7a97dd343fb40d37b` (2026-04-01, đúng bản paper dùng). Yêu cầu `python>=3.10`, `transformers>=4.56.0`.
* KVPress nén cache **trong lúc prefill phần ngữ cảnh**, rồi mới đưa câu hỏi vào. Press có hàm `compress(module, hidden_states, keys, values, attentions, kwargs) -> (keys, values)`. `ScorerPress` chấm điểm bằng `score()`, rồi giữ `topk`.
* VECTOR = một press **bọc ngoài** `KeyDiffPress` / `SnapKVPress`: dùng `score()` của press gốc, rồi thực hiện 3 bước ở 4.2.

### 4.2. Pipeline VECTOR (paper mục 3.2)
1. **Budget Relaxation:** giữ top $(1 - p_c + p_a)$ token theo điểm của base scorer.
2. **Residual Evaluation:** $\varepsilon_i = \|V_i - W_{\text{OLS}}K_i\|_2^2$, với $K$ đã khử RoPE.
3. **Asymmetric Truncation:** giữ $K$ của toàn bộ tập ứng viên; $2p_a$ token có $\varepsilon$ nhỏ nhất thì bỏ $V$ và dùng $\hat V = W_{\text{OLS}}K$; $1 - p_c - p_a$ token còn lại giữ cả $K$ và $V$.
* **Ngân sách:** $1 - p_c + p_a$ key cộng $1 - p_c - p_a$ value, bằng $1 - p_c$ cặp KV.
* **$p_a$** (Eq. 1): $\min(p_c/2,\ (1 - p_c - \epsilon)/2)$, tức $\{0.125, 0.25, 0.125, 0.05\}$ cho $p_c = \{0.25, 0.50, 0.75, 0.90\}$.
* **Calibration:** C4, chuỗi 4,096 token. Paper dùng 10,000 chuỗi để fit và 100 chuỗi held-out. Nếu dùng ít hơn, ghi rõ số token thực dùng.
* **Chế độ hiện thực:** *chất lượng* (tái hiện): trả về $V$ với token Approximate được thay bằng $\hat V$, bộ nhớ tính theo công thức. *Bộ nhớ vật lý* (mở rộng): cache chỉ lưu $V$ của tầng Retain; $\hat V$ dựng tạm thời rồi bỏ, không giữ lại.

## 5. ĐÁNH GIÁ

| Khía cạnh | Bài toán | Chỉ số | Đối chiếu |
| :--- | :--- | :--- | :--- |
| Hiệu chuẩn | C4 held-out | $R^2$ theo layer, $R^2_{global}$ | paper Table 1: 0.9392 |
| Hiểu văn bản dài | LongBench (benchmark của KVPress), tập con nhiệm vụ ghi rõ | chỉ số chính thức, trung bình | paper Table 3 |
| Độ nhạy $p_a$ | 2 nhiệm vụ LongBench, $p_c = 0.75$ | điểm theo $p_a$ | paper Figure 3 |
| Truy xuất | NIAH của KVPress (PaulGrahamEssays, ROUGE-L), $p_c = 0.90$, lưới thu nhỏ | heatmap | paper Figure 4 |
| Bộ nhớ (mở rộng) | đo trên GPU | bộ nhớ KV lâu dài (chính), Peak VRAM (phụ) | VECTOR không vượt baseline ở cùng $p_c$ |

* **Phương pháp:** No compression; KeyDiff; KeyDiff + VECTOR; SnapKV (`query_aware` như paper); SnapKV + VECTOR.
* Seed 42. dtype: bf16 trên GPU hỗ trợ; trên T4 phải kiểm tra NaN trước (đã gặp NaN với fp16 + eager ở Qwen2.5).

## 6. LỘ TRÌNH
Chi tiết ở `tasks/roadmap.md`: khảo sát KVPress → môi trường → baseline so với paper → calibration → VECTOR press → tái hiện LongBench / độ nhạy / NIAH → chuẩn hóa bằng chứng → mở rộng → báo cáo.

## 7. ĐẦU RA
1. Báo cáo đồ án: lý thuyết, phương pháp, **bảng tái hiện so với paper**, thảo luận sai lệch.
2. Mã nguồn: calibration $W_{\text{OLS}}$; VECTOR press; script chạy LongBench / NIAH; README tái lập.
3. Biểu đồ: bảng so với paper, $R^2$ theo layer, đường độ nhạy $p_a$, heatmap NIAH.
