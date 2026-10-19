# Kế Hoạch Đánh Giá & Tái Cấu Trúc Đồ Án VECTOR KV Cache (Project III)

## 1. Tổng Quan (Overview)
Đề tài: **"Nghiên cứu và Hiện thực Phương pháp Nén KV Cache Bất đối xứng Dựa trên Tính Cộng tuyến K–V cho Mô hình Ngôn ngữ Nhỏ (SLMs)"**.
Mục tiêu của kế hoạch này là rà soát toàn diện từ đặc tả kỹ thuật (Specification), kiến trúc mã nguồn hiện tại (`src/`), kết quả thực nghiệm sơ bộ (`reports/`), chỉ ra nguyên nhân gốc rễ của các lỗi hiện tại (Throughput thấp 3.9 tokens/s, NIAH thất bại), và tái cấu trúc hệ thống đạt đúng các tiêu chuẩn khoa học và kỹ thuật.

---

## 2. Đánh Giá Hiện Trạng: Hướng Đi Có Quá Đơn Giản, Quá Phức Tạp Hay Đúng Đắn?

### 2.1. Về tính đúng đắn (Methodological & Theoretical Soundness): **RẤT ĐÚNG ĐẮN VỀ Ý TƯỞNG, NHƯNG LỖI NGHIÊM TRỌNG Ở KHÂU HIỆN THỰC**
- **Điểm mạnh (Lý thuyết chuẩn xác):**
  - Khái niệm **Asymmetric Sensitivity** (Key nằm trong Softmax nhạy cảm hơn Value) và **K-V Collinearity** ($V \approx W_{\text{OLS}} K$) là nền tảng khoa học vững chắc từ các nghiên cứu hàng đầu năm 2024.
  - Cơ chế **Budget Swap** ($N_K + N_R = 2 \times \text{target\_budget}$) đã được chứng minh và pass unit test về mặt bảo toàn dung lượng phần tử bộ nhớ.
  - Tính chất Training-Free (chỉ cần Offline OLS Calibration 1 lần trên C4/Wikitext) rất phù hợp với phần cứng sinh viên và khuôn khổ Project III.
- **Tử huyệt ở khâu hiện thực (Root Causes of Failures):**
  1. **Thiếu Attention Sink (Sink Tokens):** Cả H2O baseline và VECTOR hiện tại chỉ giữ lại `recent_tokens` mà KHÔNG khóa cố định 4–32 tokens đầu tiên (`initial_tokens`). Trong kiến trúc Transformer giải mã, Softmax bắt buộc dồn phân phối chú ý vào các token đầu tiên (StreamingLLM / H2O paper). Khi thiếu sink tokens, phân phối chú ý bị vỡ, dẫn tới hiện tượng mô hình sinh ra chuỗi ký tự rác và trượt 100% bài kiểm tra NIAH.
  2. **Nghẽn cổ chai Throughput (3.9 tokens/s thay vì 35-50 tokens/s):**
     - Tại mỗi decode step, mã nguồn gọi lại hàm `_compute_reconstruction_error` tính lại sai số $\epsilon_i$ cho hàng trăm candidate tokens. Trong khi đó, vector $K_i, V_i$ của token quá khứ là bất biến, $\epsilon_i$ chỉ cần tính 1 lần duy nhất!
     - Mã nguồn dùng vòng lặp Python lồng nhau (`for b in range... for h in range...`) và gọi `torch.nonzero()`, gây 56 lần đồng bộ CPU-GPU (host-device synchronization) ở mỗi token sinh ra qua 28 layers!
  3. **Nguy cơ OOM khi dùng `output_attentions=True`:**
     - Module `VectorAttentionHookManager` bật cờ `output_attentions=True` trên Hugging Face. Ở ngữ cảnh 8K–16K, ma trận attention kích thước $[B, H, L, L]$ sẽ ngốn từ 3GB đến 14GB VRAM chỉ riêng cho activation attention weights, làm mất đi hoàn toàn ý nghĩa của việc nén KV Cache!

### 2.2. Về độ phức tạp (Complexity Balance):
- **Không quá đơn giản:** Dự án chạm đến các tầng sâu của Transformer (RoPE decoupling, GQA attention hook, custom dynamic cache kế thừa Hugging Face `DynamicCache`, OLS covariance accumulation).
- **Vừa over-engineered vừa under-engineered ở từng phần:**
  - *Over-engineered:* Tái phân vùng 3 trạng thái và tính lại sai số OLS liên tục ở mọi token decode.
  - *Under-engineered:* Bỏ qua attention sinks, bỏ qua caching sai số $\epsilon_i$, hook attention phụ thuộc hoàn toàn vào cơ chế eager attention của Hugging Face mà chưa có cơ chế fallback nhẹ nhàng.

---

## 3. Kế Hoạch Triển Khai Nhiệm Vụ (Task List)

### Giai đoạn 1: Sửa Lỗi Tử Huyệt Thuật Toán & Ổn Định Suy Luận (Fix Sinks & Throughput)
- [x] **Task 1.1: Bổ sung Cơ Chế Attention Sink (Initial Tokens) cho Baseline H2O và VECTOR**
  - Lưu giữ cố định `num_sink_tokens` (mặc định 4 hoặc 8 tokens) không bao giờ bị evict.
  - Kiểm chứng: Perplexity phục hồi, không còn sinh ký tự vô nghĩa khi nén.
- [x] **Task 1.2: Tối Ưu Hóa Tốc Độ Decode (Vectorization & Error Caching)**
  - Vector hóa logic phân vùng, loại bỏ vòng lặp `for b in ... for h in ...` và `torch.nonzero()`.
  - Cache lại $\epsilon_i$ của các token đã tính, decode step chỉ cần tính $\epsilon$ cho token mới.
  - Kiểm chứng: Throughput giải mã trên GPU tăng từ ~3.9 t/s lên $\ge 25\text{--}35$ t/s.

### Giai đoạn 2: Tái Đánh Giá Thực Nghiệm Cốt Lõi (Core Verification)
- [ ] **Task 2.1: Chạy lại Single-Needle NIAH trên H2O và VECTOR**
  - Chạy lại bài test cây kim ở độ dài 2048, 4096 tokens với các mức nén $50\%, 75\%$.
  - Xác nhận VECTOR lấy lại cây kim thành công (Pass rate $\ge 90\%$) và vượt trội H2O ở các độ sâu giữa.
- [ ] **Task 2.2: Tối ưu Hóa Attention Capture (Giảm Peak Memory Prefill)**
  - Thay vì `output_attentions=True` toàn chuỗi gây ngốn hàng GB VRAM, thiết kế cơ chế hook chỉ lưu slice attention cuối hoặc giảm thiểu allocation.

### Giai đoạn 3: Hoàn Thiện Thực Nghiệm Mở Rộng & Đóng Gói (Evaluation & Deliverables)
- [ ] **Task 3.1: Benchmark Multi-Needle NIAH & Memory Footprint**
  - Đánh giá khả năng truy xuất đa kim (3-5 needles) để chứng minh giá trị của Budget Swap.
- [ ] **Task 3.2: Đóng gói Báo cáo và Soạn thảo Đồ án**
  - Cập nhật số liệu thực nghiệm chuẩn xác vào báo cáo Project III.
