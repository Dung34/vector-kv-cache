# GIAO THỨC BẰNG CHỨNG (áp dụng cho MỌI báo cáo, ưu tiên cao hơn mọi chỉ dẫn khác)

## A. Nguyên tắc
Một kết quả chỉ tồn tại nếu nó nằm trong file log do một lệnh thật sinh ra trong phiên này.
Không có log = không có kết quả. Không được "mô tả điều sẽ xảy ra", "dựng lại output",
hay "tóm tắt từ trí nhớ". Cấm điền số liệu theo kỳ vọng.

## B. Ba nhãn bắt buộc
Mỗi câu khẳng định trong báo cáo phải mang đúng một nhãn:
- [QUAN SÁT] kèm đường dẫn file log + số dòng chứa bằng chứng.
- [SUY LUẬN] kèm tiền đề đã quan sát mà nó suy ra từ đó.
- [CHƯA KIỂM CHỨNG] nghĩa là chưa chạy gì cả; nêu lệnh sẽ chạy để kiểm.
Câu không có nhãn bị coi là không hợp lệ. Số tốc độ, số VRAM, số dòng code, tên hàm
trong thư viện đều chỉ được gắn [QUAN SÁT].

## C. Mọi thí nghiệm là một file script, chạy qua tee
- Script đặt ở experiments/phaseN_tên.py. Chạy đúng dạng:
    python experiments/phaseN_tên.py --nonce <SỐ> 2>&1 | tee logs/phaseN_tên.log
  <SỐ> do người dùng đưa. Script phải in nonce ở dòng đầu. Nonce là cách tôi
  chứng minh bạn vừa chạy thật.
- Dòng đầu log luôn là header do script tự in: nonce, timestamp, python/torch/
  transformers version, device, seed, sha256 của file script.
- Báo cáo phải kèm: đường dẫn script, đường dẫn log, exit code, sha256 của log
  (sha256sum logs/*.log).
- Dán output vào báo cáo bằng cách cat nguyên file log, không chỉnh sửa, không cắt
  dòng giữa. Nếu quá dài, dán đầu + cuối và ghi rõ "đã cắt dòng X-Y, xem log".

## D. Giá trị phải được TÍNH, không được viết sẵn
- Mọi shape, position, độ dài được lấy từ tensor/hook thật tại thời điểm chạy
  (register_forward_hook, hoặc wrap hàm bằng monkeypatch rồi ghi vào list).
  Cấm gõ tay số liệu vào print.
- Cấm print("PASS") / print("OK") cố định. Kết quả pass/fail phải đến từ assert
  hoặc phép so sánh được tính, in cả hai vế: "expected=<giá trị tính> actual=<giá trị tính>".
- Bảng trong báo cáo (bước | seen_tokens | physical_len | mask_kv_len | position_id |
  scores_len) phải do script in ra từ dữ liệu đã ghi. Cấm tự vẽ bảng.

## E. Khẳng định về thư viện phải có bằng chứng nguồn
- Muốn nói "hàm X làm gì" hoặc "dòng N của file F": dán output của
  inspect.getsource(X) hoặc `sed -n 'N1,N2p' F` hoặc `grep -n`. Không trích số dòng từ trí nhớ.
- Muốn nói "bản transformers A hành xử khác bản B": phải cài và chạy cả hai, hoặc ghi
  [CHƯA KIỂM CHỨNG]. Cấm đoán hành vi của bản chưa chạy.

## F. Bằng chứng kiểm thử hai chiều (bắt buộc cho mọi test)
Một test chỉ có giá trị nếu có CẢ BA log:
1. Chạy trên code cũ: test FAIL (dán log, thông báo assert phải chỉ đúng bất biến vi phạm).
2. Chạy trên code đã sửa: test PASS.
3. Kiểm tra đột biến: cố ý phá một dòng của bản sửa (ví dụ get_seq_length trả
   physical_len), test phải quay lại FAIL. Nếu không FAIL thì test vô dụng, báo cáo.
Thiếu một trong ba thì không được tuyên bố "đã sửa".

## G. Trung thực khi kết quả không như ý
- Nếu không chạy được (thiếu môi trường, lỗi cài đặt, hết thời gian): ghi
  "KHÔNG CHẠY ĐƯỢC: <lý do>" kèm lỗi thật. Cấm mô phỏng kết quả thay thế.
- Nếu output khác dự đoán của bạn: báo kết quả khác đó ĐẦU TIÊN, không biện giải cho
  khớp giả thuyết. Hãy viết: "Giả thuyết H bị bác bỏ vì <dòng log>".
- Nếu output có điểm bất thường (ví dụ in ra Layer 1 trong khi lẽ ra đã crash ở
  Layer 0), phải giải thích bằng một thí nghiệm, không bằng lập luận.
- Cấm các cụm: "chắc chắn", "100%", "hoàn toàn", "đã xác nhận" khi chưa có log đi kèm.

## H. Quy tắc chặn vòng lặp
- Cùng một hướng sửa thất bại 2 lần: DỪNG, nộp log, chờ tôi.
- Không sửa nhiều thứ cùng lúc. Mỗi lần sửa chỉ đổi MỘT hành vi, rồi chạy lại toàn bộ test.
- Không sửa file test để test pass. Nếu cho rằng test sai, giải thích và chờ tôi duyệt.

## I. Mẫu báo cáo mỗi phase (đúng cấu trúc này)
1. Các lệnh đã chạy (liệt kê nguyên văn).
2. File script + sha256; file log + sha256; exit code.
3. Output log (nguyên văn).
4. Bảng "Khẳng định | Nhãn | Bằng chứng (file:dòng)".
5. Những điều CHƯA làm hoặc CHƯA kiểm chứng.
6. Điều bất ngờ hoặc kết quả trái dự đoán.
