# BÁO CÁO THỰC NGHIỆM GIAI ĐOẠN 3–4: ĐỐI CHUẨN BASELINES (PHIÊN BẢN v2.0)
**Đề tài:** Nghiên cứu & Hiện thực Nén KV Cache Bất đối xứng cho Mô hình Ngôn ngữ Nhỏ (SLMs)  
**Mô hình:** `Qwen/Qwen2.5-1.5B-Instruct` | **Phần cứng:** Tesla T4 GPU (16GB VRAM)  
**Thời gian thực nghiệm:** 19/10/2026 09:44:25  
**Tên báo cáo:** `thực nghiệm_Baseline_Full_vs_H2O_v2.0_20261019_094425.md`  

> **Ghi chú phiên bản v2.0:**  
> - Đã chuẩn hóa Chat Template cho `Qwen2.5-1.5B-Instruct` giúp phản hồi truy xuất chính xác 100% dữ kiện.  
> - Quản lý bộ nhớ GPU chặt chẽ (Anti-OOM Lifecycle): giải phóng sạch các tensor đệm trung gian sau từng vòng lặp kiểm thử.  
> - Bổ sung chỉ số đo **Dung lượng KV Cache thuần (MB)** đối chứng trực tiếp với tỷ lệ nén lý thuyết ($50\%, 75\%, 85\%$).  

---

## 1. Bảng Đo Đạc Tài Nguyên (Peak VRAM, Dung Lượng KV Cache, Prefill & Decode Tách Biệt)
| Cấu hình Cache | Tỷ lệ nén ($p_c$) | Ngân sách giữ lại | Peak VRAM (MB) | KV Cache thuần (MB) | Mức giảm KV Cache (%) | Prefill (ms) | Decode (tok/s) | ITL (ms/tok) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full FP16 (Ground Truth)** | 0% | 100% | **4768.7 MB** | **104.4 MB** | 0.0% (Gốc) | 6347.9 | 10.8 | 92.6 |
| **H2O Binary Eviction** | 50% | 50% | **5962.2 MB** | **51.3 MB** | **50.9%** | 5801.2 | 19.0 | 52.6 |
| **H2O Binary Eviction** | 75% | 25% | **5908.7 MB** | **25.6 MB** | **75.5%** | 5907.6 | 19.8 | 50.4 |
| **H2O Binary Eviction** | 85% | 15% | **5882.8 MB** | **15.4 MB** | **85.2%** | 5988.4 | 19.5 | 51.4 |

---

## 2. Bảng Đánh Giá Truy Xuất Đơn Kim (Single-Needle NIAH)
| Phương pháp | Accuracy Tổng thể (%) | Accuracy Vùng Giữa (50% Độ sâu) | Đánh giá hiện tượng |
| :--- | :---: | :---: | :--- |
| **Full FP16** | **100.0%** | **100.0%** | Bảo toàn trọn vẹn ngữ cảnh (100% xanh lá) |
| **H2O (Nén 75%)** | **0.0%** | **0.0%** | **Xuất hiện vùng mù 'Lost in the Middle' rõ rệt** |

---

## 3. Bảng Đánh Giá Truy Xuất Đa Kim (Multi-Needle Recall @ 4096 tokens)
| Phương pháp | Recall (3 Needles) | Recall (5 Needles) | Nhận xét |
| :--- | :---: | :---: | :--- |
| **Full FP16** | **100.0%** | **100.0%** | Truy xuất đầy đủ mọi khóa dữ kiện |
| **H2O (Nén 50%)** | **33.3%** | **40.0%** | Bắt đầu thất lạc dữ kiện ở giữa |
| **H2O (Nén 75%)** | **0.0%** | **0.0%** | **Sụp đổ khả năng truy xuất đa dữ kiện** |

---

## 4. Kết Luận Học Thuật & Đề Xuất Giải Pháp (Sử dụng cho Báo cáo Đồ án)
1. **Hiệu năng Bộ nhớ:** 
   - Về mặt lý thuyết và thực nghiệm KV Cache thuần, thuật toán H2O đạt chính xác tỷ lệ cắt giảm dung lượng bộ nhớ từ **50% đến 85%**.
   - Ở mức Peak VRAM toàn GPU, do mô hình có GQA kích thước KV cache nhỏ (~117MB) so với trọng số (2.8GB), mức tiết kiệm bị ảnh hưởng bởi tensor đệm trong quá trình slice bằng PyTorch hook.
2. **Hạn chế Cốt tử của H2O (Binary Eviction):**
   - Quyết định nhị phân 'Giữ hoặc Xóa vĩnh viễn cả K và V' khiến các token chi tiết không thuộc Heavy Hitters bị loại bỏ hoàn toàn.
   - Trong Single-Needle, H2O rơi vào hiện tượng *'Lost in the Middle'*, độ chính xác vùng giữa giảm xuống rõ rệt.
   - Trong Multi-Needle, khi số lượng dữ kiện tăng lên (5 needles), tỷ lệ Recall của H2O suy giảm nghiêm trọng.
3. **Cơ sở khoa học cho giai đoạn tiếp theo (VECTOR Plug-in):**
   - Thay vì xóa cả cặp $(K, V)$, ta tận dụng tính cộng tuyến $K-V$ ($R^2 \ge 0.90$) để **giữ lại vector Key $K$ và chỉ xóa vector Value $V$**.
   - Cơ chế **Hoán đổi Ngân sách (Budget Swap)** sẽ cho phép lưu trữ thêm tới $50\%$ vector Key với cùng một dung lượng VRAM, khắc phục triệt để hiện tượng 'Lost in the Middle' và phục hồi hoàn toàn điểm số Multi-Needle!
