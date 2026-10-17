# R-KV: Nén KV Cache Nhận Biết Độ Dư Thừa Cho Các Mô Hình Suy Luận (Reasoning Models)

**Tác giả:** Zefan Cai¹, Wen Xiao²*, Hanshi Sun³, Cheng Luo⁴, Yikai Zhang¹, Ke Wan⁵, Yucheng Li⁶, Yeyang Zhou⁵, Li-Wen Chang, Jiuxiang Gu⁷, Zhen Dong⁸, Anima Anandkumar⁴, Abedelkadir Asi², Junjie Hu¹*  
*¹Đại học Wisconsin - Madison, ²Microsoft, ³Đại học Carnegie Mellon, ⁴Học viện Công nghệ California (Caltech), ⁵Đại học California - San Diego, ⁶Đại học Surrey, ⁷Adobe, ⁸Đại học California - Berkeley*  
*\*Tác giả liên hệ: Wen Xiao (wxiao@microsoft.com), Junjie Hu (junjie.hu@wisc.edu)*  
*Dự án & Mã nguồn:* [https://zefan-cai.github.io/R-KV.page/](https://zefan-cai.github.io/R-KV.page/) | [GitHub R-KV](https://github.com/Zefan-Cai/R-KV) | [KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory)  
*Bản thảo Preprint đang phản biện (NeurIPS).*

---

## Mục lục
1. [Tóm tắt (Abstract)](#tóm-tắt-abstract)
2. [1. Giới thiệu (Introduction)](#1-giới-thiệu-introduction)
3. [2. Các quan sát thực nghiệm (Observations)](#2-các-quan-sát-thực-nghiệm-observations)
   - [2.1 Hiện tượng dư thừa trong các mô hình suy luận](#21-hiện-tượng-dư-thừa-trong-các-mô-hình-suy-luận-redundancy-in-reasoning-models)
   - [2.2 Sự thất bại của các phương pháp nén KV Cache hiện có khi xử lý độ dư thừa](#22-sự-thất-bại-của-các-phương-pháp-nén-kv-cache-hiện-có-khi-xử-lý-độ-dư-thừa)
4. [3. Phương pháp R-KV (Redundancy-aware KV Cache Compression)](#3-phương-pháp-r-kv-redundancy-aware-kv-cache-compression)
   - [3.1 Nén tại thời điểm giải mã (Decoding-time Compression)](#31-nén-tại-thời-điểm-giải-mã-decoding-time-compression)
   - [3.2 Đánh giá tầm quan trọng qua trọng số chú ý (Importance Scoring via Attention Weights)](#32-đánh-giá-tầm-quan-trọng-qua-trọng-số-chú-ý-importance-scoring-via-attention-weights)
   - [3.3 Ước lượng độ dư thừa qua độ tương đồng ngữ nghĩa (Redundancy Estimation via Semantic Similarity)](#33-ước-lượng-độ-dư-thừa-qua-độ-tương-đồng-ngữ-nghĩa-redundancy-estimation-via-semantic-similarity)
   - [3.4 Chiến lược chọn lọc kết hợp để duy trì KV Cache (Joint Selection Strategy for KV Cache Retention)](#34-chiến-lược-chọn-lọc-kết-hợp-để-duy-trì-kv-cache-joint-selection-strategy-for-kv-cache-retention)
5. [4. Thực nghiệm (Experiments)](#4-thực-nghiệm-experiments)
   - [4.1 Thiết lập thực nghiệm (Experimental Setup)](#41-thiết-lập-thực-nghiệm-experimental-setup)
   - [4.2 Kết quả đạt được (Results)](#42-kết-quả-đạt-được-results)
6. [5. Thảo luận chuyên sâu (Discussion)](#5-thảo-luận-chuyên-sâu-discussion)
   - [5.1 Lựa chọn siêu tham số $\lambda$ như thế nào? (How to Choose $\lambda$?)](#51-lựa-chọn-siêu-tham-số-lambda-như-thế-nào-how-to-choose-lambda)
   - [5.2 Phân tích nguyên nhân thất bại của các phương pháp thuần Attention](#52-phân-tích-nguyên-nhân-thất-bại-của-các-phương-pháp-thuần-attention)
   - [5.3 Phân tích hiệu quả bộ nhớ và tính toán (Efficiency Analysis)](#53-phân-tích-hiệu-quả-bộ-nhớ-và-tính-toán-efficiency-analysis)
7. [6. Các công trình liên quan (Related Work)](#6-các-công-trình-liên-quan-related-work)
8. [7. Kết luận (Conclusion)](#7-kết-luận-conclusion)
9. [Phụ lục A: Chi tiết phương pháp và cài đặt (Method & Implementation Details)](#phụ-lục-a-chi-tiết-phương-pháp-và-cài-đặt)
   - [Thuật toán 1: Mã giả quy trình nén R-KV](#thuật-toán-1-quy-trình-nén-r-kv-algorithm-1)
   - [Chi tiết kỹ thuật: Max-Pooling cho GQA & Hiệu chuẩn Mask SnapKV](#chi-tiết-kỹ-thuật-bổ-sung)
10. [Phụ lục B: Thiết bị và Kết quả thực nghiệm chi tiết](#phụ-lục-b-thiết-bị-và-kết-quả-thực-nghiệm-chi-tiết)
11. [Phụ lục C: Phân tích độ phức tạp bộ nhớ, tính toán và thông lượng](#phụ-lục-c-phân-tích-độ-phức-tạp-bộ-nhớ-tính-toán-và-thông-lượng)
12. [Phụ lục D: Các hạn chế của nghiên cứu (Limitations)](#phụ-lục-d-các-hạn-chế-của-nghiên-cứu-limitations)

---

## Tóm tắt (Abstract)

Các mô hình suy luận (reasoning models) đã thể hiện hiệu năng vô cùng ấn tượng trong việc tự phản tư (self-reflection) và lập luận theo chuỗi tư duy (Chain-of-Thought - CoT). Tuy nhiên, chúng thường sinh ra chuỗi đầu ra dài quá mức, dẫn đến kích thước bộ nhớ đệm khóa-giá trị (Key-Value / KV Cache) phình to đến mức không thể duy trì nổi trong quá trình suy luận (inference). 

Mặc dù suy luận CoT cải thiện đáng kể độ chính xác ở các tác vụ phức tạp, nhưng việc triển khai chúng cùng với các phương pháp nén KV Cache hiện có lại thường dẫn đến các thất bại suy luận nghiêm trọng. Để giải quyết nút thắt này, chúng tôi đề xuất **R-KV (Redundancy-aware KV Cache Compression)** — phương pháp nén KV Cache nhận biết độ dư thừa dành riêng cho các mô hình suy luận, nhắm trực tiếp vào việc loại bỏ các token dư thừa. 

Phương pháp của chúng tôi:
- **Bảo toàn gần như 100% hiệu năng** so với việc dùng toàn bộ KV Cache (Full KV) trong khi **chỉ sử dụng 10% dung lượng bộ nhớ KV Cache**, vượt trội rõ rệt so với các phương pháp nén hiện tại (vốn chỉ giữ được khoảng 60% hiệu năng).
- Đáng kinh ngạc, trên tập dữ liệu AIME-24 với mô hình DeepSeek-R1-Distill-Llama-8B, R-KV thậm chí **đạt 105% độ chính xác** so với mức cơ sở Full KV khi chỉ sử dụng khoảng 16% KV Cache.
- Việc cắt giảm KV Cache này mang lại khả năng **tiết kiệm tới 90% bộ nhớ** và **tăng tốc thông lượng (throughput) lên tới 6.6×** so với quá trình suy luận CoT tiêu chuẩn.
- Các kết quả thực nghiệm trên hai tập dữ liệu suy luận toán học lớn cho thấy R-KV liên tục vượt trội hơn tất cả các phương pháp nén KV Cache cơ sở.

---

## 1. Giới thiệu (Introduction)

Những tiến bộ vượt bậc gần đây của các mô hình ngôn ngữ lớn (LLM) đã chứng minh năng lực đáng kinh ngạc trong việc lập luận phức tạp và tự phản tư. Tuy nhiên, các mô hình suy luận hiện đại (tiêu biểu như DeepSeek-R1 [1]) bộc lộ một thách thức chí mạng khi triển khai thực tế: xu hướng sinh ra các chuỗi suy luận dài dòng và lặp lại quá mức dẫn đến nhu cầu bộ nhớ không thể đáp ứng được [2], chủ yếu xuất phát từ sự phình to nhanh chóng của bộ nhớ đệm Key-Value (KV Cache) trong quá trình sinh từ tự hồi quy (autoregressive generation). 

> **Ví dụ điển hình:** Mô hình DeepSeek-R1-Distill-Llama-8B có thể sinh tới 32.000 token để giải quyết một bài toán phức tạp; quá trình này tiêu tốn 15.5 GB bộ nhớ GPU để chứa trọng số mô hình và mất thêm 4.1 GB chỉ để lưu trữ KV Cache cho một luồng giải mã đơn lẻ. Xu hướng sinh chuỗi tư duy dài (long CoT) bắt buộc chúng ta phải phát triển các giải pháp nén KV Cache hiệu quả.

Đầu ra của các mô hình suy luận hiện nay, đặc biệt trong các tác vụ CoT phức tạp, mang một đặc tính căn bản là **sự dư thừa lan tỏa (pervasive redundancy)**. Bản chất này khiến chuỗi đầu ra chứa đầy các nội dung thừa thãi:
- Những bước tự phản tư không cần thiết,
- Các phép tính toán trung gian được lặp đi lặp lại nhiều lần,
- Những đoạn độc thoại nội tâm dài dòng.

Tất cả những nội dung này hầu như không bổ sung thêm giá trị ngữ nghĩa mới nào, nhưng lại kéo dài độ dài chuỗi sinh vượt xa mức cần thiết cho một quá trình lập luận ngắn gọn, hiệu quả. Phân tích của chúng tôi (§2.1) chỉ ra rằng **hơn một nửa số token trong chuỗi suy luận của R1 đóng góp rất ít vào hiệu năng tác vụ cuối cùng**. Điều này hàm ý rằng các bước tự xác minh lặp lại hoặc các phép tính trung gian hoàn toàn có thể được cô đọng đáng kể bằng các kỹ thuật nén KV Cache mà không làm tổn hại đến độ chính xác lập luận.

Tuy nhiên, các công trình nén KV Cache hiện có [3, 4, 5, 6, 7] chủ yếu giải quyết ngữ cảnh đầu vào dài (long input prompts trong giai đoạn prefilling), chứ chưa từng được nghiên cứu sâu cho chuỗi đầu ra sinh dài (long generation outputs trong giai đoạn decoding). Hơn thế nữa, dựa trên các quan sát thực nghiệm của chúng tôi (§2.2), các phương pháp nén KV Cache tiêu chuẩn dựa trên việc lọc tầm quan trọng theo trọng số chú ý (attention-based importance) thường **thất bại nặng nề**:
- Các phân đoạn lặp lại có xu hướng tự tạo ra tín hiệu chú ý (attention signals) rất cao cho chính bản thân chúng.
- Việc loại bỏ ngây thơ các token có "trọng số chú ý thấp" có thể vô tình xóa bỏ những mảnh ghép lập luận quan trọng nhưng nằm rải rác, hoặc ngược lại, giữ lại quá mức các đoạn tự phản tư trùng lặp chỉ vì chúng có điểm attention ảo rất cao.

Quan sát này là động lực thúc đẩy chúng tôi nghiên cứu chiến lược nén nhận biết độ dư thừa (redundancy-aware compression), nhằm giữ lại một cách có chọn lọc những ngữ cảnh *"vừa quan trọng, vừa không trùng lặp"* trong suốt quá trình giải mã để bảo toàn năng lực suy luận cốt lõi của mô hình.

Trong công trình này, chúng tôi đề xuất **R-KV (Redundancy-aware KV Cache Compression)** dành cho các mô hình suy luận. Phương pháp gồm 3 trụ cột kỹ thuật:
1. **Cơ chế chấm điểm quan trọng dựa trên trọng số chú ý (Attention-based Importance Scoring):** Lựa chọn các token then chốt cần giữ lại (§3.2).
2. **Cơ chế chấm điểm dư thừa động (Dynamic Redundancy Scoring):** Phát hiện các token trùng lặp thông qua việc phân tích trực tiếp các vector Key trong thời gian thực (§3.3).
3. **Cơ chế loại bỏ kết hợp (Joint Eviction Mechanism):** Cân bằng đồng thời giữa độ dư thừa và tầm quan trọng để tối ưu hóa hiệu quả bộ nhớ cache (§3.4).

Trong các thực nghiệm trên các bài toán suy luận toán học chuẩn (§4):
- Bằng cách chỉ giữ lại **10% – 34%** dung lượng KV Cache gốc, R-KV đạt hiệu năng tương đương hoàn toàn với mô hình không nén (Full KV), vượt xa các baseline tiên tiến vốn chỉ giữ được 60% hiệu năng.
- Đáng kinh ngạc, R-KV đạt tới **105% độ chính xác của Full KV baseline** với chỉ khoảng **16% KV cache** khi chạy mô hình DeepSeek-R1-Distill-Llama-8B trên tập AIME-24.

Tiến bộ này giải quyết mâu thuẫn căn bản trong việc triển khai các LLM tiên tiến nhất: cân bằng giữa năng lực suy luận mạnh mẽ và giới hạn tài nguyên bộ nhớ thực tế. Đóng góp của chúng tôi cung cấp bằng chứng có hệ thống rằng sự dư thừa trong quá trình sinh CoT hoàn toàn có thể được nén có chiến lược. Là một phương pháp **không cần huấn luyện (training-free)** và **độc lập với kiến trúc (model-agnostic)**, R-KV có thể được ứng dụng trực tiếp ngay trong quá trình sinh mẫu (rollout) của học tăng cường (RL) cũng như trong các hệ thống phục vụ mô hình (LLM serving) quy mô lớn.

---

## 2. Các quan sát thực nghiệm (Observations)

### 2.1 Hiện tượng dư thừa trong các mô hình suy luận (Redundancy in Reasoning Models)

Như đã được ghi nhận trong nghiên cứu [2], các mô hình suy luận thường tạo ra một chuỗi tư duy chi tiết cùng nhiều bước phản tư, dẫn đến câu trả lời dài hơn đáng kể so với các mô hình tiêu chuẩn. 

Hình 2 trong bài báo thể hiện rõ: trên hai tập dữ liệu suy luận toán học phổ biến (MATH-500 và AIME 2024), cả ba mô hình suy luận gồm *DeepSeek-R1-Distill-Llama-8B*, *DeepSeek-R1-Distill-Qwen-7B* và *DeepSeek-R1-Distill-Qwen-14B* đều sinh ra độ dài chuỗi trả lời **dài hơn gấp 8 lần** so với đáp án chuẩn (Ground Truth).

Tuy nhiên, không phải tất cả các token bổ sung đều đóng góp nội dung có ý nghĩa; phần lớn ngữ cảnh được giải mã bị chi phối bởi sự lặp lại:
- Tần suất trung bình của các cụm từ 1-gram và 2-gram trong chuỗi sinh của các mô hình suy luận luôn cao vượt trội so với đáp án chuẩn (cao hơn từ **5 đến 7 lần**).
- Điều này chứng minh định lượng sự tồn tại của mức độ trùng lặp từ ngữ và lặp lại ý tưởng rất cao trong đầu ra của các mô hình suy luận.

```
So sánh độ dài sinh và tần suất lặp n-gram trên MATH-500 & AIME 2024:
+ MATH-500:
  - Ground Truth:     209.7 tokens  | 1-gram freq:  31.2 | 2-gram freq:  5.2
  - R1-Qwen-7B:     2,932.0 tokens  | 1-gram freq: 197.7 | 2-gram freq: 15.5
  - R1-Llama-8B:    2,979.0 tokens  | 1-gram freq: 214.4 | 2-gram freq: 16.6
  - R1-Qwen-14B:    2,833.0 tokens  | 1-gram freq: 167.6 | 2-gram freq: 13.3
+ AIME 2024:
  - Ground Truth:   1,547.0 tokens  | 1-gram freq:  17.2 | 2-gram freq:  3.5
  - R1-Qwen-7B:    14,592.0 tokens  | 1-gram freq:  81.9 | 2-gram freq: 10.2
  - R1-Llama-8B:   15,535.8 tokens  | 1-gram freq:  98.3 | 2-gram freq: 11.7
  - R1-Qwen-14B:   12,402.0 tokens  | 1-gram freq:  84.1 | 2-gram freq:  9.8
```

---

### 2.2 Sự thất bại của các phương pháp nén KV Cache hiện có khi xử lý độ dư thừa

Hầu hết các phương pháp nén KV Cache hiện tại ưu tiên lựa chọn token dựa trên mức độ quan trọng theo ngữ cảnh, thường được đo bằng điểm chú ý (attention scores) giữa token key và token query [3, 4]. Mặc dù cách tiếp cận này giữ lại ngữ cảnh quan trọng khá tốt trong văn bản thông thường, nhưng nó **hoàn toàn thất bại trước tính dư thừa đặc thù của các mô hình suy luận**.

Trong các mô hình suy luận, chúng tôi phát hiện ra rằng **nội dung lặp lại thường nhận được điểm chú ý cao một cách bất thường**, bởi vì nó phản chiếu rất sát các đoạn văn bản trùng lặp đã được tạo ra trước đó. Hậu quả là các token dư thừa bị giữ lại quá mức trong bộ nhớ cache, làm phình to kích thước KV Cache một cách vô ích mà không đem lại bất kỳ thông tin mới nào.

#### Ví dụ minh họa thực tế (Case Study - Trích xuất từ Hình 3):
Một bài toán mẫu: *"Trong lớp học gồm 30 học sinh của thầy Roper..."*
- Mô hình DeepSeek-R1 giải bài toán với chuỗi suy luận gồm:
  - Phần đề bài và chỉ dẫn: 102 từ.
  - Đoạn suy nghĩ ban đầu: 203 từ.
  - **Quá trình tự phản tư lặp đi lặp lại 13 lần với tổng cộng 581 từ!**
  - Các đoạn lặp xuất hiện liên tục:
    - *"Nhưng khoan đã, vậy 10% của 30 là 3. Vậy có 3 học sinh về sớm."* (36 từ)
    - *"Nhưng trong bài toán ban đầu... Vậy có 3 học sinh về sớm."* (42 từ)
    - *"Nhưng khoan đã... 30 là 3. Vậy có 3 học sinh về sớm."* (36 từ)
    - *(Lặp lại cấu trúc này thêm 5 lần nữa!)*
    - Đoạn kết luận: *"Tôi nghĩ thế là xong. Phép tính rất rõ ràng: 10% của 30 là 3."*
- Khi áp dụng phương pháp nén dựa trên Attention (như SnapKV):
  - Do các cụm từ *"3 students are leaving early"* và *"But wait..."* xuất hiện liên tục, chúng tự kích hoạt trọng số chú ý rất cao lẫn nhau.
  - SnapKV ưu tiên giữ lại hàng loạt token của các câu lặp này trong KV Cache.
  - Các bước tính toán quan trọng khác nhưng chỉ xuất hiện một lần lại bị xóa mất vì điểm chú ý thấp hơn các cụm lặp, dẫn tới mô hình suy luận sai đáp án cuối cùng.

---

## 3. Phương pháp R-KV (Redundancy-aware KV Cache Compression)

Để giải quyết vấn đề tư duy dư thừa, chúng tôi đề xuất **R-KV** — phương pháp nén KV Cache tại thời điểm giải mã có khả năng nhận biết độ dư thừa. Phương pháp này cân bằng giữa **tầm quan trọng (importance)** và **tính không dư thừa (non-redundancy)** trong việc lựa chọn token, đảm bảo dung lượng lưu trữ KV Cache luôn được phân bổ cho các nội dung vừa giàu thông tin vừa đa dạng.

Cấu trúc tổng thể của R-KV gồm 3 giai đoạn:
1. **Nén tại thời điểm giải mã (Decoding-Time Compression - §3.1)**
2. **Chấm điểm quan trọng & Ước lượng độ dư thừa (§3.2, §3.3)**
3. **Nén KV Cache bằng chiến lược chọn lọc kết hợp (§3.4)**

---

### 3.1 Nén tại thời điểm giải mã (Decoding-time Compression)

Khác với các phương pháp nén KV Cache trước đây [3, 5, 4] chủ yếu tập trung vào giai đoạn nạp prompt (prefilling), R-KV tập trung giải quyết giai đoạn giải mã (decoding) của mô hình suy luận — một bối cảnh đặc thù nơi đầu ra sinh ra dài hơn gấp nhiều lần đầu vào.

Cơ chế phân bổ bộ nhớ:
- **Cache lưu trữ dài hạn có ngân sách:** Kích thước cố định $B_{\text{budget}}$ để lưu các token KV được giữ lại.
- **Bộ đệm tạm thời:** Kích thước $B_{\text{buffer}}$ dùng để chứa các token văn bản mới được sinh ra trong chu kỳ hiện tại.
- Tổng bộ nhớ KV cần cấp phát: $B_{\text{total}} = B_{\text{budget}} + B_{\text{buffer}}$.

**Quy trình thực thi theo chu kỳ:**
1. Sau khi mô hình sinh được một đoạn văn bản có độ dài cố định $B_{\text{buffer}}$ vào bộ đệm, quá trình nén R-KV được kích hoạt.
2. $\alpha$ token cuối cùng của đoạn mới sinh luôn được bảo toàn trong cache để làm **các token quan sát (observation tokens)** (kế thừa nguyên lý từ SnapKV [3]).
3. Chúng ta nối $B_{\text{budget}}$ token sẵn có trong cache với $(B_{\text{buffer}} - \alpha)$ token đầu tiên trong buffer, tạo thành tập gồm:
   $$n = B_{\text{budget}} + B_{\text{buffer}} - \alpha \quad \text{token ứng viên KV (candidate tokens)}$$
4. Mỗi token ứng viên được chấm một điểm chọn lọc tổng hợp (§3.4). Ta chọn ra top $k = B_{\text{budget}} - \alpha$ token có điểm số cao nhất để đưa vào phần ngân sách cache còn lại, kết hợp cùng $\alpha$ token quan sát.
5. Quá trình này giúp nén KV cache hiệu quả mà vẫn bảo toàn ngữ cảnh cốt lõi xuyên suốt quá trình sinh tự hồi quy.

---

### 3.2 Đánh giá tầm quan trọng qua trọng số chú ý (Importance Scoring via Attention Weights)

Theo trực giác từ các phương pháp dựa trên attention, các token nhận được sự chú ý cao hơn từ các bước giải mã hiện tại sẽ đóng góp nhiều hơn vào quá trình sinh tiếp theo. R-KV tính toán điểm chú ý mà mỗi Key token nhận được từ $\alpha$ token quan sát cuối cùng. Chúng tôi xây dựng công thức hỗ trợ đầy đủ cho cả Multi-Head Attention (MHA) và Grouped-Query Attention (GQA):

#### Multi-Head Attention (MHA)
Cho $\alpha$ token quan sát làm query $Q^h \in \mathbb{R}^{\alpha \times d}$ và $n$ trạng thái key $K^h \in \mathbb{R}^{n \times d}$ tại head $h$, ma trận điểm chú ý $A^h \in \mathbb{R}^{\alpha \times n}$ được tính bằng:
$$A^h = \text{softmax}\left(\frac{Q^h (K^h)^\top}{\sqrt{d}}\right) \tag{1}$$

#### Grouped-Query Attention (GQA)
Trong kiến trúc GQA hiện đại, mỗi cặp head Key/Value được chia sẻ chung cho một nhóm gồm $G$ Query head khác nhau (đánh chỉ số $g \in [0, G)$). Ký hiệu các trạng thái Key/Value dùng chung là $K^h, V^h \in \mathbb{R}^{n \times d}$, và $G$ trạng thái Query trong nhóm là $Q^{h, 0}, \dots, Q^{h, G-1} \in \mathbb{R}^{\alpha \times d}$. Điểm chú ý của từng Query head trong nhóm được tính:
$$A^{h,g}_{\text{group}} = \frac{Q^{h,g} (K^h)^\top}{\sqrt{d}} \in \mathbb{R}^{\alpha \times n}, \quad \forall g = 0, \dots, G-1 \tag{2}$$

Khác với các nghiên cứu trước đây dùng Mean-Pooling, R-KV áp dụng phép toán **Max-Pooling** dọc theo chiều nhóm query để bảo toàn các token then chốt nhất đối với từng Query head độc lập. Sau đó, ma trận tổng hợp $A^h_{\text{group}}$ được tái chuẩn hóa bằng Softmax dọc theo chiều Key token:
$$A^h_{\text{group}} = \text{maxpool}\left(A^{h,0}_{\text{group}}, \dots, A^{h,G-1}_{\text{group}}\right) \in \mathbb{R}^{\alpha \times n}$$
$$A^h = \text{softmax}\left(A^h_{\text{group}}\right) \in \mathbb{R}^{\alpha \times n} \tag{3}$$

#### Ổn định và làm mượt điểm quan trọng (Stabilization)
Ký hiệu $A^h$ là ma trận trọng số chú ý đã tính được (dù từ MHA hay GQA). Điểm quan trọng lấy trực tiếp từ $A^h$ có thể chứa các giá trị ngoại lai (outliers) quá lớn, dẫn đến việc ước lượng kém ổn định. Để khắc phục, chúng tôi áp dụng phép Max-Pooling 1D với cửa sổ trượt kích thước $2W$ qua các token lân cận, sau đó lấy trung bình dọc theo $\alpha$ token quan sát:
$$\tilde{A}^h_{j,i} = \max\left(A^h_{j, i-W}, \dots, A^h_{j, i}, \dots, A^h_{j, i+W-1}\right)$$
$$I^h_i = \frac{1}{\alpha} \sum_{j=0}^{\alpha-1} \tilde{A}^h_{j,i} \in \mathbb{R} \tag{4}$$

Trong đó $I^h_i$ chính là **Điểm Quan Trọng (Importance Score)** của token thứ $i$ tại attention head $h$.

---

### 3.3 Ước lượng độ dư thừa qua độ tương đồng ngữ nghĩa (Redundancy Estimation via Semantic Similarity)

Để nhận diện các token dư thừa, chúng tôi đo lường độ tương đồng ngữ nghĩa giữa các trạng thái Key bằng Cosine Similarity. Những token có độ tương đồng cao với nhiều token khác được xem là có khả năng dư thừa cao và có thể lược bỏ.

#### Tính ma trận Cosine Similarity
Với các token Key $K^h \in \mathbb{R}^{n \times d}$ của head $h$, ta chuẩn hóa L2 từng vector $K^h_i$ thành $\hat{K}^h_i$ rồi tính ma trận tương đồng $S^h \in \mathbb{R}^{n \times n}$:
$$\hat{K}^h_i = \frac{K^h_i}{\|K^h_i\|_2 + \epsilon} \in \mathbb{R}^d, \quad S^h = \hat{K}^h (\hat{K}^h)^\top \in \mathbb{R}^{n \times n} \tag{5}$$
Để ngăn việc token tự tính độ dư thừa với chính nó, ta gán các phần tử trên đường chéo chính bằng 0:
$$S^h_{i,i} \leftarrow 0, \quad \forall i \in [0, n)$$

#### Cơ chế bảo vệ các token được sinh ra gần nhất (Enforce Retention of Recent Tokens)
Mặc dù bị trùng lặp, một số token lặp lại gần đây vẫn có thể chứa thông tin ngữ cảnh có giá trị định hướng cho các bước tiếp theo. Việc xóa bỏ ngây thơ toàn bộ các token tương đồng có thể làm đứt gãy mạch lập luận. Để giải quyết, **chúng tôi ưu tiên giữ lại $\beta$ token được tạo ra gần đây nhất trong số các token có độ tương đồng cao**, bởi vì các token sinh sau thường phản ánh trạng thái suy luận hiện tại tốt hơn các token sinh từ rất sớm:
- Với mỗi token $i \in [0, n)$, xác định tập các token có độ tương đồng vượt ngưỡng $T$:
  $$\mathcal{I}^h_i = \{j \mid S^h_{j,i} > T, j \in [0, n)\}$$
- Từ tập này, trích xuất tập con $\mathcal{I}^h_{i,\beta} \subseteq \mathcal{I}^h_i$ chứa tối đa $\beta$ token có chỉ số vị trí lớn nhất (tức các token xuất hiện gần nhất).
- Ta triệt tiêu liên kết tương đồng trực tiếp của chúng bằng cách gán điểm tương đồng về 0:
  $$S^h_{j,i} \leftarrow 0, \quad \forall j \in \mathcal{I}^h_{i,\beta}$$

#### Ước lượng điểm dư thừa (Redundancy Score Estimation)
Đầu tiên, với mỗi token $i$ tại head $h$, ta tính độ tương đồng trung bình của nó với toàn bộ các token khác:
$$\bar{S}^h_i = \frac{1}{n} \sum_{j=0}^{n-1} S^h_{j,i}$$
Một giá trị $\bar{S}^h_i$ lớn chứng tỏ nội dung ngữ nghĩa của token $i$ bị chia sẻ lặp lại rất nhiều trong chuỗi. Tiếp theo, để đưa điểm số về thang đo chuẩn mực, ta áp dụng hàm Softmax:
$$R^h_i = \left(\text{softmax}\left[\bar{S}^h_0, \dots, \bar{S}^h_{n-1}\right]\right)_i \tag{6}$$

Trong đó $R^h_i$ là **Điểm Dư Thừa (Redundancy Score)** của token thứ $i$ tại head $h$. Giá trị này càng cao phản ánh mức độ trùng lặp càng lớn.

---

### 3.4 Chiến lược chọn lọc kết hợp để duy trì KV Cache (Joint Selection Strategy for KV Cache Retention)

Nhằm tối ưu hóa không gian lưu trữ mà vẫn giữ trọn ngữ cảnh thiết yếu, chúng tôi áp dụng chiến lược kết hợp cả hai chỉ số quan trọng và dư thừa. Với một ngân sách token $B_{\text{budget}}$ định trước cho mỗi head, điểm chọn lọc tổng hợp $Z^h_i$ cho token $i$ tại head $h$ được tính:
$$Z^h_i = \lambda I^h_i - (1 - \lambda) R^h_i \tag{7}$$

- $I^h_i$: Điểm quan trọng từ phương trình (4).
- $R^h_i$: Điểm dư thừa từ phương trình (6).
- $\lambda \in [0, 1]$: Siêu tham số điều chỉnh sự đánh đổi giữa việc ưu tiên token quan trọng và việc giảm thiểu token trùng lặp.
- Điểm được lấy trung bình trên tất cả các head: $\text{AggScore}_k = \frac{1}{H} \sum_{h=0}^{H-1} Z^h_k$.
- Chọn top $B_{\text{budget}} - \alpha$ token có $\text{AggScore}$ cao nhất để đưa vào cache dài hạn cùng với $\alpha$ token quan sát.

---

## 4. Thực nghiệm (Experiments)

### 4.1 Thiết lập thực nghiệm (Experimental Setup)

- **Mô hình nghiên cứu:** Sử dụng các phiên bản mô hình chưng cất từ DeepSeek-R1 gồm `DeepSeek-R1-Distill-Llama-8B` (viết tắt là R1-Llama-8B) và `DeepSeek-R1-Distill-Qwen-14B` (R1-Qwen-14B).
- **Bộ dữ liệu chuẩn:** MATH-500 và AIME 2024.
- **Siêu tham số mặc định:** $B_{\text{buffer}} = 128$, $\alpha = 8$, $\lambda = 0.1$, ngưỡng tương đồng $T = 0.8$, số lượng token gần nhất được bảo vệ $\beta = 4$.
- **Phương pháp đối chứng (Baselines):**
  - **SnapKV [3]:** Phương pháp nén dựa trên Attention tiên tiến nhất. Để so sánh công bằng trên giai đoạn decoding, chúng tôi áp dụng cùng chu kỳ nén 128 bước sinh một lần với cùng ngân sách $B_{\text{budget}}$ và $B_{\text{buffer}}$.
  - **FullKV:** Giữ lại toàn bộ KV Cache không nén, đóng vai trò là "chuẩn vàng" (gold standard) về chất lượng giải mã.
- **Quy trình đánh giá:** Độ dài sinh tối đa được đặt là 16,384 token cho MATH-500 và 32,768 token cho AIME 2024. Chúng tôi áp dụng phép đánh giá **Pass@1** qua **64 câu trả lời được sinh độc lập** cho mỗi câu hỏi (sử dụng sampling temperature = 0.6 và top-p = 0.95 khuyến nghị của DeepSeek-R1). Công thức tính: $\text{Pass@1} = \frac{1}{k} \sum_{i=1}^k p_i$ (với $p_i$ là tính đúng sai của câu trả lời thứ $i$).

---

### 4.2 Kết quả đạt được (Results)

Hiệu năng độ chính xác (Pass@1, %) của R-KV so với SnapKV và FullKV được trình bày chi tiết dưới đây:

#### Bảng kết quả độ chính xác (Pass@1, %) trên MATH-500 và AIME 2024:

| Mô hình | Dataset | Phương pháp | Ngân sách KV Cache (Tokens) | | | | | | | | |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | | | **128** | **256** | **512** | **768** | **1,024** | **1,536** | **2,048** | **2,560** | **3,072** |
| **R1-Llama-8B** | **MATH-500** | FullKV | 82.38 | 82.38 | 82.38 | 82.38 | 82.38 | 82.38 | 82.38 | – | – |
| | | **R-KV (Ours)** | **51.08** | **67.39** | **76.92** | **80.21** | **81.34** | **82.34** | **82.65** | – | – |
| | | SnapKV | 32.53 | 50.07 | 64.03 | 70.81 | 74.43 | 78.43 | 80.50 | – | – |
| | **AIME 2024** | FullKV | 49.79 | 49.79 | 49.79 | 49.79 | 49.79 | 49.79 | 49.79 | 49.79 | 49.79 |
| | | **R-KV (Ours)** | **0.42** | **10.21** | **29.48** | **40.31** | **45.26** | **51.56** | **52.29** | **53.85** | **53.13** |
| | | SnapKV | 0.16 | 0.94 | 4.53 | 11.20 | 15.73 | 26.04 | 32.76 | 39.43 | 41.93 |
| **R1-Qwen-14B** | **MATH-500** | FullKV | 94.58 | 94.58 | 94.58 | 94.58 | 94.58 | 94.58 | 94.58 | – | – |
| | | **R-KV (Ours)** | **56.21** | **73.33** | **84.77** | **88.79** | **90.72** | **92.72** | **93.62** | – | – |
| | | SnapKV | 26.32 | 43.93 | 77.93 | 82.52 | 86.63 | 90.86 | 92.73 | – | – |
| | **AIME 2024** | FullKV | 65.68 | 65.68 | 65.68 | 65.68 | 65.68 | 65.68 | 65.68 | – | 65.68 |
| | | **R-KV (Ours)** | **0.57** | **7.92** | **24.53** | **36.25** | **42.66** | **55.00** | **56.09** | – | **64.32** |
| | | SnapKV | 0.26 | 2.86 | 12.86 | 16.30 | 25.00 | 36.41 | 46.56 | – | 52.86 |

#### Phân tích theo hai góc nhìn ngân sách:

1. **Góc nhìn Tỷ lệ Ngân sách (Ratio Budget):**
   - Với `R1-Llama-8B` (độ dài sinh TB 2,979 trên MATH và 15,535 trên AIME), R-KV đạt trạng thái **nén không suy hao (lossless compression)** chỉ với **34% ngân sách cache** trên MATH-500 và **10% ngân sách cache** trên AIME 2024.
   - Khi được cấp 16% ngân sách cache trên AIME 2024, R-KV đạt độ chính xác **52.29%**, vượt qua cả FullKV (49.79%) — đạt **105% hiệu năng**.
   - Với `R1-Qwen-14B` (độ dài sinh TB 2,833 trên MATH và 12,402 trên AIME), R-KV đạt lossless với 54% cache trên MATH và 25% cache trên AIME. Khi cấp 33% cache trên AIME, R-KV đạt 67.45%, tương đương **105% hiệu năng FullKV**.

2. **Góc nhìn Ngân sách Cố định (Fixed Budget):**
   - Đây là thiết lập mang tính thực tế cao nhất khi triển khai: khi chuỗi đầu ra kéo dài từ 2,979 token (MATH) lên 15,535 token (AIME), dung lượng KV cache cần thiết để đạt mức nén không suy hao tăng lên rất ít (chỉ cần thêm khoảng 512 token).
   - Với `R1-Llama-8B`, R-KV đạt lossless với ngân sách cố định 1,024 token trên MATH-500 và 1,536 token trên AIME 2024.
   - Với `R1-Qwen-14B`, R-KV đạt lossless với ngân sách cố định 1,536 token trên MATH-500 và 3,072 token trên AIME 2024.

---

## 5. Thảo luận chuyên sâu (Discussion)

### 5.1 Lựa chọn siêu tham số $\lambda$ như thế nào? (How to Choose $\lambda$?)

Hình 5 của bài báo trực quan hóa phân phối của Điểm Quan Trọng ($I^h$) và Điểm Dư Thừa ($R^h$) tại head $h=0$ ở layer trên cùng ($N_{\text{layer}} = 31$):
- **Đặc tính phân phối:** Điểm $I^h$ rất thưa thớt (sparse) và bị chi phối bởi một số ít giá trị ngoại lai cực lớn, trong khi phân phối độ tương đồng tạo nên $R^h$ lại tương đối dày đặc (dense).
- **Khi $\lambda = 0$:** Chiến lược giữ token bị chi phối hoàn toàn bởi điểm dư thừa $R^h$. Các token khởi đầu (initial 4 tokens / sink tokens) không được đảm bảo giữ lại. Nghiên cứu [7] đã chỉ ra rằng việc xóa các token khởi đầu này sẽ phá hủy hoàn toàn khả năng sinh từ của LLM, dẫn đến sụp đổ hiệu năng. Do đó, $\lambda$ bắt buộc phải lớn hơn 0 (khuyến nghị $\ge 0.01$).
- **Khi $\lambda > 0.1$:** Điểm Attention bắt đầu lấn át hoàn toàn điểm tương đồng, khiến mô hình lại rơi vào bẫy giữ lại các đoạn lặp lại.
- **Kết luận:** Khoảng giá trị tối ưu là $0.01 \le \lambda \le 0.1$. Thực nghiệm trên mô hình Llama-8B với tập dữ liệu MATH-500 (Hình 6) khẳng định rằng **$\lambda = 0.1$ cho độ chính xác cao nhất**. Cả hai thái cực chỉ dựa vào dư thừa ($\lambda=0$) hoặc chỉ dựa vào attention ($\lambda=1$) đều cho kết quả tồi tệ nhất, chứng minh tính chất bổ trợ thiết yếu của hai đại lượng này.

---

### 5.2 Phân tích nguyên nhân thất bại của các phương pháp thuần Attention

Để làm rõ ưu thế vượt trội của cơ chế kết hợp giữa Attention và Redundancy, chúng tôi so sánh tập các token được giữ lại bởi R-KV và SnapKV trong một trường hợp cụ thể mà R-KV trả lời đúng còn SnapKV trả lời sai (Hình 7):

#### Chi tiết bài toán và quá trình suy luận (Hình 7):
- **Đề bài:** Lớp học có 30 học sinh. 20% học sinh tham gia đội bóng bầu dục về sớm. Trong số học sinh còn lại, 25% là đội cổ vũ hoặc ban nhạc cũng về sớm. Hỏi có bao nhiêu học sinh về sớm?
- **Cách SnapKV lựa chọn token (Hình 7a):**
  - SnapKV bị thiên lệch mạnh về các token nằm gần token query hiện tại.
  - Các cụm từ lặp lại như *"3 students are leaving early"* và *"But in the initial..."* được chọn nhiều lần bởi rất nhiều attention head (hiển thị màu đỏ đậm).
  - Kết quả là SnapKV giữ lại quá nhiều token dư thừa không liên quan, làm mất các bước tính toán trừ học sinh ban đầu ($30 - 6 = 24$), dẫn đến kết luận sai rằng có 3 học sinh về sớm.
- **Cách R-KV lựa chọn token (Hình 7b):**
  - Nhờ bị trừ điểm bởi $R^h$, các phân đoạn lặp lại bị phân bổ điểm thấp hơn và bị loại bỏ bớt.
  - R-KV lựa chọn một tập hợp token đa dạng, trải rộng đồng đều trên toàn bộ chuỗi giải mã.
  - R-KV bảo toàn trọn vẹn các bước tính toán: $20\% \times 30 = 6$, phần còn lại là $30 - 6 = 24$, tiếp tục $25\% \times 24 = 6$, và tổng cộng là $6 + 6 = 12$ học sinh. Mô hình đưa ra kết luận chính xác là 12 học sinh.

---

### 5.3 Phân tích hiệu quả bộ nhớ và tính toán (Efficiency Analysis)

#### Bảng so sánh Tiết kiệm Bộ nhớ, Thông lượng và Thời gian Giải mã trên Llama3-8B (GPU NVIDIA A100 80GB):

| Độ dài sinh | Phương pháp | Thiết lập Ngân sách | Tiết kiệm Bộ nhớ (%) | Batch Size tối đa | Thông lượng (tok/s) | Tổng số token sinh | Thời gian giải mã (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **8K** | **FullKV** | – | – | 1 | 75.44 | 8,094 | 107.30 |
| | | – | – | **62** (max) | **849.13** | 501,828 | 590.99 |
| | **SnapKV** | Cố định – 1,024 | 87.50% | 402 (max) | 3,253.93 | 3,253,788 | 999.96 |
| | | Cố định – 1,536 | 81.25% | 287 (max) | 2,525.25 | 2,322,978 | 919.90 |
| | | Tỷ lệ – 10% (819) | 90.00% | 479 (max) | 3,808.81 | 3,877,026 | 1,017.91 |
| | **R-KV (Ours)** | Cố định – 1,024 | **87.50%** | 1 | 80.46 | 8,094 | 100.60 |
| | | Cố định – 1,024 | **87.50%** | **402** (max) | **3,251.52** | 3,253,788 | 1,000.70 |
| | | Cố định – 1,536 | 81.25% | 287 (max) | 2,525.75 | 6,546,972 | 919.72 |
| | | Cố định – 3,072 | 62.50% | 150 (max) | 1,520.99 | 1,214,100 | 798.23 |
| | | Tỷ lệ – 10% (819) | **90.00%** | **479** (max) | **3,809.15** | 3,877,026 | 1,017.82 |
| | | Tỷ lệ – 34% (2,785)| 66.00% | 167 (max) | 1,608.01 | 1,351,698 | 840.61 |
| | | Tỷ lệ – 54% (4,423)| 46.00% | 105 (max) | 1,257.83 | 849,870 | 675.66 |
| **16K** | **FullKV** | – | – | 1 | 69.41 | 16,286 | 234.65 |
| | | – | – | **30** (max) | **347.03** | 488,580 | 1,407.89 |
| | **SnapKV** | Cố định – 1,024 | 93.75% | 402 (max) | 3,202.17 | 6,546,972 | 2,044.54 |
| | | Tỷ lệ – 10% (1,638)| 90.00% | 271 (max) | 2,306.26 | 4,413,506 | 1,913.71 |
| | **R-KV (Ours)** | Cố định – 1,024 | **93.75%** | 1 | 80.95 | 16,286 | 201.18 |
| | | Cố định – 1,024 | **93.75%** | **402** (max) | **3,188.82** | 6,546,972 | 2,053.10 |
| | | Cố định – 1,536 | 90.63% | 287 (max) | 2,447.61 | 4,674,082 | 1,909.65 |
| | | Tỷ lệ – 10% (1,638)| **90.00%** | **271** (max) | **2,300.28** | 4,413,506 | 1,918.68 |
| | | Tỷ lệ – 34% (5,570)| 66.00% | 82 (max) | 797.43 | 1,335,452 | 1,674.70 |
| | | Tỷ lệ – 54% (8,847)| 46.00% | 46 (max) | 584.77 | 749,156 | 1,281.12 |

#### Phân tích chi tiết:
1. **Tiết kiệm bộ nhớ (Memory Saving):** Bộ nhớ của R-KV được ấn định kích thước cố định thay vì tăng tuyến tính theo chiều dài chuỗi sinh như FullKV. Ở chuỗi sinh 16K, R-KV tiết kiệm tới **93.75%** bộ nhớ KV Cache.
2. **Chi phí tính toán phụ trội (Computation Overhead):** Mặc dù R-KV có thêm bước tính ma trận Cosine Similarity $\mathcal{O}(B_{\text{budget}}^2)$, chi phí này rất nhỏ và được bù đắp hoàn toàn bởi việc giảm khối lượng tính toán Self-Attention trên toàn bộ KV Cache nén. Khi batch size = 1, R-KV thậm chí giải mã nhanh hơn FullKV (80.95 vs 69.41 tok/s ở độ dài 16K).
3. **Bùng nổ thông lượng hệ thống (End-to-End Throughput):** Lợi ích lớn nhất của R-KV là giải phóng VRAM, cho phép tăng vọt kích thước Batch Size:
   - Ở chuỗi dài 16K với ngân sách 1,024, Batch Size tối đa tăng từ **30 lên 402** (tăng **13.4×**), đẩy thông lượng tổng từ 347 tok/s lên **3,188 tok/s (tăng 9.2×)**!
   - So với SnapKV, chi phí tính thêm độ dư thừa của R-KV chỉ làm giảm thông lượng chưa tới 1%, nhưng lại mang lại độ chính xác vượt trội hơn từ 10% đến 40%.

---

## 6. Các công trình liên quan (Related Work)

1. **Nén KV Cache (KV Cache Compression):**
   - *Lược bỏ token động (Token Eviction):* SnapKV [3], PyramidKV [5], Ada-KV [22], HeadKV [6] tập trung loại bỏ token trong giai đoạn prefilling ngữ cảnh dài. StreamingLLM [7] và H2O [4] áp dụng cho decoding. Tuy nhiên, các kỹ thuật tổng quát này gặp khó khăn nghiêm trọng trong các tác vụ suy luận nặng, nơi việc lược bỏ token bừa bãi sẽ phá hủy các bước trung gian trong CoT.
   - *Lượng tử hóa (Quantization) [13, 14, 15], Hợp nhất token (Merging) [16, 17, 18], Phân rã hạng thấp (Low-rank decomposition) [19, 20, 21]:* Là các hướng tiếp cận trực giao có thể kết hợp song song với R-KV để tối ưu sâu hơn.

2. **Suy luận hiệu quả (Efficient Reasoning):**
   - *Huấn luyện ép ngắn CoT:* Sử dụng Reinforcement Learning với phần thưởng phạt độ dài [23, 24, 25] hoặc dùng tập dữ liệu CoT biến thiên độ dài để Supervised Fine-Tuning (SFT) [26, 27]. Nhược điểm lớn: Tốn kém chi phí tính toán huấn luyện lại mô hình.
   - *Prompting tại thời điểm kiểm tra (Test-time prompting) [27, 28, 29]:* Dùng prompt yêu cầu mô hình suy nghĩ ngắn gọn. Nhược điểm: Thường làm sụt giảm trực tiếp độ chính xác của các bài toán khó.
   - **Ưu thế của R-KV:** Là phương pháp nén KV Cache không cần huấn luyện lại, không cần sửa đổi prompt, đạt trạng thái nén không suy hao trực tiếp trong quá trình serving.

---

## 7. Kết luận (Conclusion)

Chúng tôi đã giới thiệu **R-KV**, phương pháp nén KV Cache tại thời điểm giải mã đầu tiên được thiết kế chuyên biệt để vượt qua thách thức về chuỗi tư duy dài và dư thừa trong các mô hình suy luận lớn (LLM). Bằng cách kết hợp giữa việc chấm điểm quan trọng dựa trên Attention và ước lượng độ dư thừa ngữ nghĩa dựa trên Cosine Similarity của các vector Key, R-KV giữ lại trọn vẹn các nội dung lập luận cốt lõi trong khi loại bỏ các token trùng lặp không mang thông tin mới.

Chiến lược năng động này cho phép R-KV bảo toàn gần như 100% hiệu năng mô hình gốc khi chỉ sử dụng **10% – 34%** dung lượng KV Cache, vượt xa các giải pháp nén hiện tại. Các phân tích chi tiết về thông lượng chứng minh R-KV cho phép mở rộng kích thước batch size lên tới **13.4×** và tăng tốc độ thông lượng xử lý lên tới **9.2×** trong các tình huống sinh chuỗi dài. Với thiết kế độc lập mô hình và không cần huấn luyện, R-KV là giải pháp sẵn sàng triển khai quy mô lớn, đặc biệt hữu ích trong việc tăng tốc pha sinh mẫu (rollout) của các quy trình học tăng cường (RL).

---

## Phụ lục (Appendix)

### Phụ lục A: Chi tiết phương pháp và cài đặt

#### Thuật toán 1: Quy trình nén R-KV (Algorithm 1)

Ký hiệu: $Q_{\text{obs}}$ là trạng thái query của $\alpha$ token quan sát, $(K_{\text{full}}, V_{\text{full}})$ là toàn bộ KV cache độ dài $L_{\text{full}}$.

```
Thuật toán 1: R-KV Compression Procedure
Đầu vào: (K_full, V_full), L_full, L_budget, Q_obs, alpha, B_budget, B_buffer, T, beta, lambda, epsilon, H, d_k

1: procedure R-KV((K_full, V_full), L_full, L_budget, Q_obs, alpha, B_budget, B_buffer, T, beta, lambda, epsilon, H, d_k)
2:    if L_full - L_budget < B_buffer then
3:        return (K_full, V_full)               # Chưa đủ chu kỳ buffer để kích hoạt nén
4:    end if
5:    (K_obs, V_obs)   ← alpha token cuối cùng của (K_full, V_full)
6:    (K_cand, V_cand) ← (L_full - alpha) token đầu tiên của (K_full, V_full)
7:    N_c ← L_full - alpha                     # Số lượng token ứng viên
8:    if N_c <= B_budget then
9:        return (K_full, V_full)               # Số ứng viên chưa vượt quá ngân sách
10:   end if
11:   for each head h = 0 ... H - 1 do
12:       Tính ma trận attention A^h ∈ R^{alpha x N_c} từ Q_obs^h và K_cand^h (xử lý MHA/GQA theo Eq. 1-3)
13:       for k = 0 ... N_c - 1 do
14:           I'_{k,h} ← (1 / alpha) * sum_{q=0}^{alpha-1} (A^h)_{qk}
15:       end for
16:       {I_{k,h}}_{k=0}^{N_c-1} ← 1D-Pooling({I'_{k,h}}_{k=0}^{N_c-1})  # Làm mượt bằng max-pooling 1D
17:   end for
18:   for each head h = 0 ... H - 1 do
19:       Kh_norm ∈ R^{N_c x d_k}; Kh_norm,k ← Kh_cand,k / (||Kh_cand,k||_2 + epsilon)
20:       S^h ← Kh_norm * (Kh_norm)^T           # Tính ma trận Cosine Similarity S^h ∈ R^{N_c x N_c}
21:       for k = 0 ... N_c - 1 do
22:           (S^h)_{kk} ← 0                    # Khử tự tương đồng trên đường chéo
23:       end for
24:       B^h_{uv} ← ((S^h)_{uv} > T ? 1 : 0)   # Đánh dấu các cặp có độ tương đồng vượt ngưỡng T
25:       for u = 0 ... N_c - 1 do
26:           T^h_u ← {v | B^h_{uv} = 1, v ∈ {0, ..., N_c - 1}}
27:           T^h_{u,beta} ← tập con của T^h_u chứa tối đa beta chỉ số v lớn nhất (sinh gần nhất)
28:           for v' in T^h_{u,beta} do
29:               (S^h)_{u,v'} ← 0              # Bảo vệ beta token gần nhất bằng cách gán 0
30:           end for
31:       end for
32:       Đặt S_bar^h ∈ R^{N_c} với (S_bar^h)_u ← (1 / N_c) * sum_{v=0}^{N_c-1} (S^h)_{uv}
33:       for u = 0 ... N_c - 1 do
34:           R_{u,h} ← (softmax(S_bar^h))_u    # Điểm dư thừa của token u tại head h
35:       end for
36:   end for
37:   for each head h = 0 ... H - 1 do
38:       for k = 0 ... N_c - 1 do
39:           Score_{k,h} ← lambda * I_{k,h} - (1 - lambda) * R_{k,h}
40:       end for
41:   end for
42:   for k = 0 ... N_c - 1 do
43:       AggScore_k ← mean_h(Score_{k,h})      # Lấy trung bình điểm trên tất cả các head
44:   end for
45:   Idx_sel ← chỉ số của top (B_budget - alpha) token có AggScore cao nhất
46:   Kcand_sel ← K_cand[Idx_sel]; Vcand_sel ← V_cand[Idx_sel]
47:   K_comp ← concatenate(Kcand_sel, K_obs)
48:   V_comp ← concatenate(Vcand_sel, V_obs)
49:   return (K_comp, V_comp)
50: end procedure
```

#### Chi tiết kỹ thuật bổ sung:
1. **Max Pooling trong Grouped-Query Attention (GQA):**  
   Các mô hình nguồn mở hiện đại phổ biến sử dụng GQA để chia sẻ head Key/Value giữa nhiều head Query. Khi nén KV cache, cần gom điểm attention từ `(Q_head, seq_len, seq_len)` về `(KV_head, seq_len, seq_len)`. SnapKV sử dụng Mean-Pooling (lấy trung bình), điều này làm loãng các token tối quan trọng của từng Query head riêng lẻ. R-KV sử dụng **Max-Pooling**, giúp bảo toàn nguyên vẹn token quan trọng nhất cho từng Query head, mang lại hiệu năng cao hơn rõ rệt.
2. **Hiệu chuẩn Mask cửa sổ quan sát (Observation Window Mask Calibration):**  
   Cài đặt chính thức của SnapKV áp dụng causal mask tam giác trên rồi mới gọi softmax, sau đó cắt lát (slicing) và tính tổng. Cách làm này khiến các token trong cửa sổ quan sát hấp thụ một phần trọng số chú ý đáng lẽ thuộc về các token tiền tố, phá vỡ tính chất chuẩn hóa tổng bằng 1. R-KV khắc phục bằng cách **cắt lát ma trận trọng số trước khi áp dụng Softmax**, đảm bảo chuẩn hóa toán học chính xác và tạo ra điểm số chất lượng hơn.

---

### Phụ lục B: Thiết bị và Kết quả thực nghiệm chi tiết

- **Phần cứng:** Toàn bộ thực nghiệm được tiến hành trên cụm máy chủ trang bị GPU **NVIDIA A100 80GB SXM**.
- **Kết quả chi tiết:** Bảng 2 trong bài báo thể hiện đầy đủ điểm Pass@1 (%) của Llama3-8B và Qwen-14B trên MATH-500 và AIME 2024 ở các mốc ngân sách từ 128 đến 4,096 tokens (đã trình bày chi tiết tại Bảng ở Mục 4.2).

---

### Phụ lục C: Phân tích độ phức tạp bộ nhớ, tính toán và thông lượng

#### C.1 Phân tích độ phức tạp bộ nhớ và tính toán
- **Bộ nhớ lưu trữ (Memory Footprint):**
  - Ngân sách lưu trữ cache cố định: $M_{\text{budget}} \in \mathbb{R}^{b \times B_{\text{budget}} \times N_{\text{layer}} \times N_{\text{head}} \times d}$
  - Bộ đệm chứa token mới sinh: $M_{\text{buffer}} \in \mathbb{R}^{b \times B_{\text{buffer}} \times N_{\text{layer}} \times N_{\text{head}} \times d}$
  - Trọng số mô hình: $M_\theta$
  - Query cache cho $\alpha$ token quan sát: $M_\alpha \in \mathbb{R}^{b \times \alpha \times N_{\text{layer}} \times N_{\text{head}} \times d}$
  - Tổng bộ nhớ R-KV: $M_{\text{total}} = M_\theta + M_{\text{budget}} + M_{\text{buffer}} + M_\alpha$
  - Lượng bộ nhớ tiết kiệm so với FullKV (với $B_{\text{full}}$ tokens):
    $$M_{\text{saving}} = M_{\text{full}} - M_{\text{budget}} - M_{\text{buffer}} - M_\alpha$$
- **Độ phức tạp tính toán (Computational Complexity):**
  - Chấm điểm quan trọng: $\mathcal{O}(\alpha \cdot B_{\text{budget}})$
  - Ước lượng độ dư thừa: $\mathcal{O}(B_{\text{budget}}^2)$
  - Tổng overhead mỗi chu kỳ nén: $\mathcal{O}(\alpha B_{\text{budget}} + B_{\text{budget}}^2)$
  - Độ phức tạp tính attention thông thường của FullKV là $\mathcal{O}(B_{\text{full}} B_{\text{buffer}})$. Khi $B_{\text{full}}$ rất lớn (10K–32K), việc đưa về $\mathcal{O}((B_{\text{budget}} + B_{\text{buffer}}) B_{\text{buffer}})$ giúp giảm tải lượng tính toán khổng lồ, bù đắp hoàn toàn chi phí tính thêm của ma trận tương đồng.

#### C.2 Phân tích thông lượng chi tiết
- **Theo ngân sách tỷ lệ (Ratio Budget):** Với chuỗi dài 16K, ở tỷ lệ nén 10%, R-KV cho phép tăng kích thước Batch Size lên gấp **9.0×** và tăng thông lượng lên **6.6×** so với FullKV. Ở các batch size nhỏ (<128), thông lượng tăng tuyến tính theo kích thước batch. Ở batch size lớn, sự tăng trưởng chậm lại do GPU A100 chuyển sang trạng thái nghẽn tính toán (compute-bound).
- **Theo ngân sách cố định (Fixed Budget):** Khi chuỗi sinh dài 16K và cố định $B_{\text{budget}} = 1024$, R-KV tăng batch size lên **13.4×** và tăng throughput lên **9.19×**. Ưu thế của ngân sách cố định càng thể hiện vượt trội khi chuỗi sinh càng dài.

---

### Phụ lục D: Các hạn chế của nghiên cứu (Limitations)

1. **Khả năng tương thích với PagedAttention:**  
   Một hạn chế hiện tại của phương pháp R-KV là khả năng tương thích với các cơ chế quản lý bộ nhớ trang tiên tiến như PagedAttention (đang dùng phổ biến trong vLLM). Việc tích hợp giải thuật nén động với cấu trúc bảng trang phi tuần tự đòi hỏi các nghiên cứu chuyên sâu tiếp theo.
2. **Yêu cầu hỗ trợ giao diện nén native trong Serving Framework:**  
   Trong các framework phục vụ suy luận hiện tại chưa có sẵn API chuyên dụng cho việc nén KV Cache, hệ thống buộc phải thực hiện việc tái cấp phát bộ nhớ (reallocating memory) và giải phóng bộ nhớ cũ. Thao tác này có thể phát sinh chi phí phụ (overhead) làm giảm một phần lợi ích tăng tốc. Khi các serving engine tích hợp giao diện KV cache compression ở mức nhân tính toán (native kernel level), hiệu năng của R-KV sẽ còn được giải phóng mạnh mẽ hơn nữa.
