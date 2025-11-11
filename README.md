# NesPDHG: Halpern-typed Methods for Linear Programming

[cite_start]Đây là kho chứa code cho bài báo nghiên cứu: **"Halpern-typed methods for LPs"**[cite: 1].

[cite_start]Nghiên cứu này tìm hiểu các phương pháp kiểu Halpern để giải các bài toán quy hoạch tuyến tính (LPs)[cite: 3]. [cite_start]Code trong kho chứa này bao gồm việc triển khai các thuật toán được mô tả, cũng như các thử nghiệm số để so sánh chúng với các phương pháp bậc nhất (first-order methods) hiện đại khác[cite: 4].

Code này được xây dựng dựa trên thư viện **MPAX** (Mathematical Programming in JAX) (bản gốc: `https://github.com/MIT-Lu-Lab/MPAX`).

## 📄 Bài báo liên quan

* [cite_start]**Tên bài báo:** Halpern-typed methods for LPs [cite: 1]
* [cite_start]**Tác giả:** Vu Thi Huong, **Le Duc Hiep**, và Thorsten Koch [cite: 2]
* [cite_start]**Ngày:** 11 tháng 11 năm 2025 [cite: 2]

> **Tóm tắt (Abstract):** In this work, we study Halpern-typed methods to solve linear programs. [cite_start]Theoretical guarantees for the convergence and convergence rates of the methods are revised, and numerical experiments to compare with state-of-the-art first-order methods are presented. [cite: 3, 4]

## 🚀 Các thuật toán được triển khai

Kho chứa này mở rộng thư viện `MPAX` gốc với các thuật toán sau:

* [cite_start]**`nesPDHG`**: Phương pháp kiểu Halpern được đề xuất trong công trình này, dựa trên mối liên hệ với phương pháp gia tốc Nesterov[cite: 202]. [cite_start]Trong các thí nghiệm, nó được cấu hình với `w=3` và `gamma=0.75`[cite: 203].
* **`nes1_pdhg`**, **`nes2_pdhg`**: Các biến thể của `nesPDHG` với các lựa chọn tham số `w` và `gamma` khác nhau.
* [cite_start]**`r2HPDHG`**: Một biến thể "Restarted Halpern PDHG" bậc hai[cite: 205].
* [cite_start]**`rHPDHG`**: Phương pháp "Restarted Halpern PDHG" cơ sở (baseline)[cite: 204].
* [cite_start]**`r2HPDHGmpax`**: Phiên bản triển khai thực tế của `r2HPDHG` có trong thư viện `MPAX`[cite: 206].

## 🛠️ Cài đặt

1.  Clone kho chứa này:
    ```bash
    git clone [https://github.com/hiepday3324/NesPDHG.git](https://github.com/hiepday3324/NesPDHG.git)
    cd NesPDHG
    ```

2.  Cài đặt các thư viện phụ thuộc. [cite_start]Dự án sử dụng JAX và được thử nghiệm trên GPU NVIDIA RTX 4090[cite: 209].
    ```bash
    # Cài đặt các thư viện từ file requirements.txt (nếu có)
    pip install -r requirements.txt
    
    # Hoặc cài đặt thủ công các thư viện chính
    pip install jax jaxlib numpy pandas
    ```

## 📊 Tái tạo kết quả

Bạn có thể sử dụng notebook **`Compare.ipynb`** để chạy so sánh hiệu suất giữa các solver khác nhau.

Notebook này sẽ giúp tái tạo lại các kết quả được trình bày trong bài báo, so sánh thời gian giải trung bình và đường cong phân phối tích lũy thực nghiệm (ECD).

## 📈 Kết quả nổi bật

Phương pháp `nesPDHG` được đề xuất cho thấy sự cải thiện đáng kể về thời gian giải quyết trung bình so với các phương pháp cơ sở.

* **Tại độ chính xác 10⁻⁴ (Figure 1):**
    * [cite_start]`nesPDHG` đạt thời gian giải trung bình thấp nhất (khoảng 22 giây)[cite: 218].
    * [cite_start]`nesPDHG` giải được 285 instances, nhiều hơn 3 instances so với `r2HPDHG` (282)[cite: 220, 234].

* **Tại độ chính xác 10⁻⁸ (Figure 3):**
    * [cite_start]`nesPDHG` tiếp tục dẫn đầu với thời gian trung bình khoảng 63 giây[cite: 286].
    * [cite_start]`nesPDHG` giải được 268 instances, nhiều hơn 21 instances so với `r2HPDHG` (247)[cite: 289, 306, 307].



## 📚 Trích dẫn (Citation)

Nếu bạn sử dụng công trình này trong nghiên cứu của mình, vui lòng trích dẫn bài báo gốc.

```bibtex
@article{HuongHiepKoch2025,
  title   = {Halpern-typed methods for LPs},
  author  = {Vu, Thi Huong and Le, Duc Hiep and Koch, Thorsten},
  journal = {ZIB Report (ArXiv Preprint)},
  year    = {2025},
  month   = {November}
}
