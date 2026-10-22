# NHIỆM VỤ: Sửa H2OKVCache / VectorKVCache để chạy decode với Qwen2 trong HuggingFace

## 0. Quy tắc làm việc (ĐỌC TRƯỚC, bắt buộc tuân thủ)
- KHÔNG sửa code cho đến khi hoàn thành Phase 0 và báo cáo kết quả.
- KHÔNG sửa lỗi shape bằng cách slice/pad/cắt mask cho "khớp". Lệch shape là triệu chứng của một vi phạm hợp đồng. Hãy tìm vi phạm đó.
- Mỗi khi gặp lỗi mới, TRƯỚC khi sửa phải viết ra 3 dòng:
  (a) tensor nào, shape thực tế là gì (in ra, không đoán);
  (b) hàm nào tạo ra tensor đó, tại thời điểm nào trong luồng;
  (c) vì sao nó khác với tensor kia, và bất biến nào (mục 3) bị vi phạm.
- Nếu cùng một hướng sửa thất bại 2 lần: DỪNG, in diagnostic đầy đủ và báo cáo. Không thử hướng thứ 3 một cách mù quáng.
- Mọi lần lặp phải chạy trên model Qwen2 CONFIG NHỎ NGẪU NHIÊN trên CPU (2 layer, hidden 64, 4 heads, 2 kv heads, context 64, budget 50%). Chỉ chạy benchmark 4096 token thật ở Phase 4.

## 1. Vấn đề cốt lõi (mô hình tư duy)
HF giả định cache chỉ có thể NỐI THÊM token, nên "số token đã thấy" == "số token đang nằm trong cache".
H2O vi phạm giả định này: update() cắt tỉa K/V, nên hai đại lượng tách ra:
- seen_tokens: số token tuyệt đối đã xử lý. Không bao giờ giảm. Dùng cho cache_position, position_ids, RoPE.
- physical_len: độ dài K/V thực tế trong cache sau tỉa. Dùng cho kích thước attention mask.

Thứ tự trong mỗi bước decode (cần XÁC MINH trên bản đã cài, xem Phase 0):
1. Qwen2Model.forward gọi past_key_values.get_seq_length() -> tính cache_position, position_ids.
2. Qwen2Model.forward tạo causal mask với kv_len dựa trên cache (get_mask_sizes hoặc _update_causal_mask). Mask được CHỐT ở đây, TRƯỚC khi layer 0 chạy.
3. Mỗi layer: self_attn -> cache.update() (nối rồi tỉa) -> eager_attention_forward -> forward hook.
Hệ quả: mask được tính từ độ dài cache TRƯỚC update, còn K thực tế lại là độ dài SAU update (đã tỉa).

Lỗi hiện tại: attn_weights [.., 1877] + mask [.., 3757]. Không phải mask "cũ từ prefill" mà là mask mới nhưng tính sai độ dài.
Lỗi ngầm thứ hai (không báo exception): nếu get_seq_length() trả về độ dài đã tỉa thì position_ids sai (~1877 thay vì ~3757). Key cũ trong cache đã xoay RoPE theo vị trí gốc, nên decode sẽ ra rác.
Lưu ý: update() nối 1 token rồi tỉa về đúng budget B, nên ở trạng thái ổn định K sau update luôn dài B.

## 2. Phase 0: Điều tra (không sửa code)
In ra và báo cáo (kèm đường dẫn file + số dòng):
1. transformers.__version__.
2. Nguồn thực tế của Qwen2Model.forward: nó gọi gì để lấy cache_position, position_ids, và tạo mask (create_causal_mask / _update_causal_mask / get_mask_sizes)?
3. Nguồn của DynamicCache.get_seq_length và (nếu có) DynamicCache.get_mask_sizes. H2OKVCache kế thừa từ lớp nào và đã override những gì?
4. Khi output_attentions=True, attn_implementation đang là gì? Có bị fallback sang eager không?
5. Với model nhỏ, chạy 1 bước decode bằng cache H2O hiện tại, in: past_seen, cache_position, position_ids, shape mask nhận vào attention, shape K sau update, ở từng layer.
Kết luận Phase 0 phải trả lời: "Mask kv_len được lấy từ hàm nào của cache?" và "position_ids được lấy từ hàm nào?". Nếu chưa trả lời được thì không sang Phase 1.

## 3. Các bất biến (invariants) cần đảm bảo
Viết thành assert/test, chạy ở mọi bước decode:
- I1. position_id của token mới == seen_tokens trước bước đó (vị trí tuyệt đối), kể cả sau khi đã tỉa.
- I2. kv_len của mask == độ dài K SAU update() ở MỌI layer. Ổn định: B. Trước khi chạm budget: past + 1.
- I3. Mọi layer có cùng độ dài K (vì mask chỉ tính một lần). Nếu không đảm bảo được, báo cáo, đừng tự bịa workaround.
- I4. accumulated_scores[layer].shape[-1] == K[layer].shape[-2] sau update() và sau khi hook cộng điểm. Khi tỉa phải gather accumulated_scores theo CÙNG selected_indices, và điểm của token mới được thêm vào với giá trị 0 trước khi topk.
- I5. K trong cache giữ nguyên RoPE của vị trí gốc, không được re-rotate.
- I6. Với compression_ratio = 0 (không tỉa), output token greedy phải GIỐNG HỆT chạy model không cache custom.

## 4. Phase 1: Test tái hiện (trước khi sửa)
Viết test model nhỏ tái hiện đúng lỗi "size of tensor a ... b". Test phải FAIL ở bản hiện tại, vì nếu không fail được thì chưa hiểu lỗi.

## 5. Phase 2: Thiết kế sửa (thứ tự ưu tiên)
A. Ưu tiên: override trong lớp cache (sửa ở lớp cha dùng chung cho H2O và Vector):
   - self._seen_tokens tăng trong update() (chỉ ở layer_idx == 0).
   - get_seq_length() trả _seen_tokens (cho position/cache_position).
   - get_mask_sizes(cache_position, layer_idx) trả (kv_len, kv_offset) với kv_len = min(physical_len + q_len, active_budget) khi q_len == 1 và đã có budget. Chữ ký phải khớp đúng bản transformers đã cài (xem Phase 0, KHÔNG copy mù từ gợi ý này).
B. Dự phòng nếu A không khả thi với bản đã cài: truyền position_ids tường minh = [[seen_tokens]] và attention_mask=None ở bước decode (q_len=1 nên không cần mask nhân quả). Phải kiểm tra bằng thực nghiệm rằng HF không tự sinh mask lại khi nhận None; nếu có thì quay về A.
Tách rõ trong thiết kế: cái gì dùng seen_tokens, cái gì dùng physical_len. Ghi chú thành comment ngay trên hàm.

## 6. Phase 2b: Vấn đề phụ cần xử lý cùng lúc
- Prefill đang gọi output_attentions=False nên hook nhận None, accumulated_scores không được khởi tạo. Quyết định có chủ đích: (i) prefill với eager + output_attentions=True để lấy điểm, hoặc (ii) khởi tạo điểm bằng cách khác (vd. dùng cửa sổ attention cuối). Nêu rõ lựa chọn và lý do, không để mặc định.
- Thứ tự trong 1 bước decode: update() (thêm slot 0 cho token mới -> tỉa + gather điểm) -> attention -> hook cộng attn_weights (kv_len = B) vào accumulated_scores. Kiểm tra shape khớp ở bước này.

## 7. Phase 3: Nghiệm thu trên model nhỏ
1. Test I1 đến I6 đều pass trong >= 20 bước decode (đủ để vượt budget nhiều lần).
2. Test ở cả H2OKVCache và VectorKVCache.
3. In bảng: bước | seen_tokens | physical_len | mask_kv_len | position_id | scores_len. Các cột phải khớp bất biến.

## 8. Phase 4: Chạy benchmark thật
Chỉ sau khi Phase 3 pass. Báo cáo: số bước decode hoàn tất, không exception, token sinh ra có hợp lý (so sánh vài token đầu với Full-FP16).

## Định dạng báo cáo mỗi phase
(1) Đã tìm thấy/làm gì, (2) bằng chứng (output đã in), (3) bất biến nào đã xác nhận, (4) bước tiếp theo. Không sang phase sau khi phase trước chưa có bằng chứng.