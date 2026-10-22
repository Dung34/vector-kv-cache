

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Code Architecture & Modularity Standards (Project III Standard)

**Keep modules decoupled, focused, and under single-responsibility limits.**

- **File Size Limit:** Keep files $\le 250\text{--}300$ lines. If a file exceeds this and manages multiple distinct concerns, decompose it immediately into specialized submodules.
- **Layered Separation of Concerns:**
  - `Policy / Algorithmic`: Pure tensor logic (e.g. `partitioner.py`). Takes input tensors, returns indices/masks, zero framework coupling.
  - `Math / Engine`: Approximation, transforms, and equations (e.g. `reconstructor.py`, RoPE decoupling, $W_{\text{OLS}}$).
  - `Container / Wrapper`: Framework lifecycle integration (e.g. `vector_cache.py` subclassing Hugging Face `DynamicCache`). Coordinates Policy and Math.
  - `Hooks`: Interception managers (e.g. `vector_hook.py`, forward hooks).
- **Public API Invariance:**
  - Always export primary classes at package `__init__.py`. External code (scripts, notebooks, benchmarks) must never break when internal files are decomposed.
- **TDD for Submodules:** Every extracted submodule MUST have its own standalone unit test running in $< 0.1\text{s}$ on CPU with mock tensors.

## 6. Giao thức bằng chứng (Evidence Protocol - Ưu tiên cao hơn mọi chỉ dẫn khác)

### A. Nguyên tắc
Một kết quả chỉ tồn tại nếu nó nằm trong file log do một lệnh thật sinh ra trong phiên này.
Không có log = không có kết quả. Không được "mô tả điều sẽ xảy ra", "dựng lại output",
hay "tóm tắt từ trí nhớ". Cấm điền số liệu theo kỳ vọng.

### B. Ba nhãn bắt buộc
Mỗi câu khẳng định trong báo cáo phải mang đúng một nhãn:
- [QUAN SÁT] kèm đường dẫn file log + số dòng chứa bằng chứng.
- [SUY LUẬN] kèm tiền đề đã quan sát mà nó suy ra từ đó.
- [CHƯA KIỂM CHỨNG] nghĩa là chưa chạy gì cả; nêu lệnh sẽ chạy để kiểm.
Câu không có nhãn bị coi là không hợp lệ. Số tốc độ, số VRAM, số dòng code, tên hàm
trong thư viện đều chỉ được gắn [QUAN SÁT].

### C. Mọi thí nghiệm là một file script, chạy qua tee
- Script đặt ở `experiments/phaseN_tên.py`. Chạy đúng dạng:
    `python experiments/phaseN_tên.py --nonce <SỐ> 2>&1 | tee logs/phaseN_tên.log`
  `<SỐ>` do người dùng đưa. Script phải in nonce ở dòng đầu. Nonce là cách chứng minh đã chạy thật.
- Dòng đầu log luôn là header do script tự in: nonce, timestamp, python/torch/transformers version, device, seed, sha256 của file script.
- Báo cáo phải kèm: đường dẫn script, đường dẫn log, exit code, sha256 của log (`sha256sum logs/*.log`).
- Dán output vào báo cáo bằng cách cat nguyên file log, không chỉnh sửa, không cắt dòng giữa. Nếu quá dài, dán đầu + cuối và ghi rõ "đã cắt dòng X-Y, xem log".

### D. Giá trị phải được TÍNH, không được viết sẵn
- Mọi shape, position, độ dài được lấy từ tensor/hook thật tại thời điểm chạy (register_forward_hook, hoặc wrap hàm bằng monkeypatch rồi ghi vào list). Cấm gõ tay số liệu vào print.
- Cấm print("PASS") / print("OK") cố định. Kết quả pass/fail phải đến từ assert hoặc phép so sánh được tính, in cả hai vế: "expected=<giá trị tính> actual=<giá trị tính>".
- Bảng trong báo cáo (bước | seen_tokens | physical_len | mask_kv_len | position_id | scores_len) phải do script in ra từ dữ liệu đã ghi. Cấm tự vẽ bảng.

### E. Khẳng định về thư viện phải có bằng chứng nguồn
- Muốn nói "hàm X làm gì" hoặc "dòng N của file F": dán output của inspect.getsource(X) hoặc `sed -n 'N1,N2p' F` hoặc `grep -n`. Không trích số dòng từ trí nhớ.
- Muốn nói "bản transformers A hành xử khác bản B": phải cài và chạy cả hai, hoặc ghi [CHƯA KIỂM CHỨNG]. Cấm đoán hành vi của bản chưa chạy.

### F. Bằng chứng kiểm thử hai chiều (bắt buộc cho mọi test)
Một test chỉ có giá trị nếu có CẢ BA log:
1. Chạy trên code cũ: test FAIL (dán log, thông báo assert phải chỉ đúng bất biến vi phạm).
2. Chạy trên code đã sửa: test PASS.
3. Kiểm tra đột biến: cố ý phá một dòng của bản sửa (ví dụ get_seq_length trả physical_len), test phải quay lại FAIL. Nếu không FAIL thì test vô dụng, báo cáo.
Thiếu một trong ba thì không được tuyên bố "đã sửa".

### G. Trung thực khi kết quả không như ý
- Nếu không chạy được (thiếu môi trường, lỗi cài đặt, hết thời gian): ghi "KHÔNG CHẠY ĐƯỢC: <lý do>" kèm lỗi thật. Cấm mô phỏng kết quả thay thế.
- Nếu output khác dự đoán: báo kết quả khác đó ĐẦU TIÊN, không biện giải cho khớp giả thuyết. Hãy viết: "Giả thuyết H bị bác bỏ vì <dòng log>".
- Nếu output có điểm bất thường (ví dụ in ra Layer 1 trong khi lẽ ra đã crash ở Layer 0), phải giải thích bằng một thí nghiệm, không bằng lập luận.
- Cấm các cụm: "chắc chắn", "100%", "hoàn toàn", "đã xác nhận" khi chưa có log đi kèm.

### H. Quy tắc chặn vòng lặp
- Cùng một hướng sửa thất bại 2 lần: DỪNG, nộp log, chờ người dùng.
- Không sửa nhiều thứ cùng lúc. Mỗi lần sửa chỉ đổi MỘT hành vi, rồi chạy lại toàn bộ test.
- Không sửa file test để test pass. Nếu cho rằng test sai, giải thích và chờ duyệt.

### I. Mẫu báo cáo mỗi phase (đúng cấu trúc này)
1. Các lệnh đã chạy (liệt kê nguyên văn).
2. File script + sha256; file log + sha256; exit code.
3. Output log (nguyên văn).
4. Bảng "Khẳng định | Nhãn | Bằng chứng (file:dòng)".
5. Những điều CHƯA làm hoặc CHƯA kiểm chứng.
6. Điều bất ngờ hoặc kết quả trái dự đoán.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
