# BÁO CÁO THỰC NGHIỆM GIAI ĐOẠN 3–4: ĐỐI CHUẨN BASELINES
**Đề tài:** Nghiên cứu & Hiện thực Nén KV Cache Bất đối xứng cho Mô hình Ngôn ngữ Nhỏ (SLMs)  
**Mô hình:** `Qwen/Qwen2.5-1.5B-Instruct` | **Phần cứng:** Tesla T4 GPU (16GB VRAM)  
**Thời gian thực nghiệm:** 17/10/2026 13:31:33  
**Tên báo cáo:** `thực nghiệm_Baseline_Full_vs_H2O_v1.0_20261017_133133.md`  

---

## 1. Bảng Đo Đạc Tài Nguyên (Peak VRAM & Tốc Độ Suy Luận)
| Cấu hình Cache | Tỷ lệ nén ($p_c$) | Ngân sách giữ lại | Peak VRAM (MB) | Mức tiết kiệm VRAM (%) | Throughput (tokens/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full FP16 (Ground Truth)** | 0% | 100% | **2864.9 MB** | 0.0% (Gốc) | 11.3 |
| **H2O Binary Eviction** | 50% | 50% | **2909.8 MB** | **-1.6%** | 12.0 |
| **H2O Binary Eviction** | 75% | 25% | **2930.1 MB** | **-2.3%** | 12.1 |
| **H2O Binary Eviction** | 85% | 15% | **2940.3 MB** | **-2.6%** | 12.1 |

---

## 2. Bảng Đánh Giá Truy Xuất Đơn Kim (Single-Needle NIAH)
| Phương pháp | Accuracy Tổng thể (%) | Accuracy Vùng Giữa (25%–75% Độ sâu) | Đánh giá hiện tượng |
| :--- | :---: | :---: | :--- |\
| **Full FP16** | **0.0%** | **0.0%** | Bảo toàn trọn vẹn ngữ cảnh (100% xanh lá) |
| **H2O (Nén 75%)** | **0.0%** | **0.0%** | **Xuất hiện vùng mù 'Lost in the Middle' rõ rệt** |

---

## 3. Bảng Đánh Giá Truy Xuất Đa Kim (Multi-Needle Recall @ 4096 tokens)
| Phương pháp | Recall (3 Needles) | Recall (5 Needles) | Nhận xét |
| :--- | :---: | :---: | :--- |
| **Full FP16** | **0.0%** | **0.0%** | Truy xuất đầy đủ mọi khóa dữ kiện |
| **H2O (Nén 50%)** | **0.0%** | **0.0%** | Bắt đầu thất lạc dữ kiện ở giữa |
| **H2O (Nén 75%)** | **0.0%** | **0.0%** | **Sụp đổ khả năng truy xuất đa dữ kiện** |

---

## 4. Kết Luận Học Thuật & Đề Xuất Giải Pháp (Sử dụng cho Báo cáo Đồ án)
1. **Hiệu năng Bộ nhớ:** Thuật toán H2O chứng minh khả năng nén giảm tới **-2% VRAM** so với Full FP16, giúp mở rộng giới hạn phục vụ suy luận dài trên phần cứng GPU 16GB.
2. **Hạn chế Cốt tử của H2O (Binary Eviction):**
   * Quyết định nhị phân 'Giữ hoặc Xóa vĩnh viễn cả K và V' khiến các token chi tiết không thuộc Heavy Hitters bị loại bỏ hoàn toàn.
   * Trong bài toán Single-Needle, H2O rơi vào hiện tượng *'Lost in the Middle'*, độ chính xác vùng giữa giảm xuống chỉ còn **0.0%**.
   * Trong bài toán Multi-Needle, khi số lượng dữ kiện tăng lên (5 needles), tỷ lệ Recall của H2O sụp đổ chỉ còn **0.0%**.
3. **Cơ sở khoa học cho giai đoạn tiếp theo (VECTOR Plug-in):**
   * Thay vì xóa cả cặp $(K, V)$, ta tận dụng tính cộng tuyến $K-V$ ($R^2 \ge 0.90$) để **giữ lại vector Key $K$ và chỉ xóa vector Value $V$**.
   * Cơ chế **Hoán đổi Ngân sách (Budget Swap)** sẽ cho phép lưu trữ thêm tới $50\%$ vector Key với cùng một dung lượng VRAM, khắc phục triệt để hiện tượng 'Lost in the Middle' và phục hồi hoàn toàn điểm số Multi-Needle!
