# NesPDHG: Halpern-typed Methods for Linear Programming

Đây là kho chứa code cho bài báo nghiên cứu: **"Halpern-typed methods for LPs"**.

Nghiên cứu này tìm hiểu các phương pháp kiểu Halpern để giải các bài toán quy hoạch tuyến tính (LPs). Code trong kho chứa này bao gồm việc triển khai các thuật toán được mô tả, cũng như các thử nghiệm số để so sánh chúng với các phương pháp bậc nhất (first-order methods) hiện đại khác.

Code này được xây dựng dựa trên thư viện **MPAX** (Mathematical Programming in JAX) (bản gốc: `https://github.com/MIT-Lu-Lab/MPAX`).

## 📄 Bài báo liên quan (Preparing)

* **Tên bài báo:** Nesterov–Halpern Methods for LPs
* **Tác giả:** Vu Thi Huong, **Le Duc Hiep**, và Thorsten Koch

> **Tóm tắt (Abstract):** In this work, we study Halpern-typed methods to solve linear programs. Theoretical guarantees for the convergence and convergence rates of the methods are revised, and numerical experiments to compare with state-of-the-art first-order methods are presented.

## 🚀 Các thuật toán được triển khai

Kho chứa này mở rộng thư viện `MPAX` gốc với các thuật toán sau:

* **`nesPDHG`**: Phương pháp kiểu Halpern được đề xuất trong công trình này, dựa trên mối liên hệ với phương pháp gia tốc Nesterov. Trong các thí nghiệm, nó được cấu hình với `w=3` và `gamma=0.75`.
* **`nes1_pdhg`**, **`nes2_pdhg`**: Các biến thể của `nesPDHG` với các lựa chọn tham số `w` và `gamma` khác nhau.
* **`r2HPDHG`**: Một biến thể "Restarted Halpern PDHG" bậc hai.
* **`rHPDHG`**: Phương pháp "Restarted Halpern PDHG" cơ sở (baseline).
* **`r2HPDHGmpax`**: Phiên bản triển khai thực tế của `r2HPDHG` có trong thư viện `MPAX`.

## 🛠️ Cài đặt

1.  Clone kho chứa này:
    ```bash
    git clone [https://github.com/hiepday3324/NesPDHG.git](https://github.com/hiepday3324/NesPDHG.git)
    cd NesPDHG
    ```

2.  Cài đặt các thư viện phụ thuộc. Dự án sử dụng JAX và được thử nghiệm trên GPU NVIDIA RTX 4090.
    ```bash
    # Cài đặt các thư viện từ file requirements.txt (nếu có)
    pip install -r requirements.txt
    
    # Hoặc cài đặt thủ công các thư viện chính
    pip install jax jaxlib numpy pandas
    ```

## 📊 Tái tạo kết quả

Notebook này sẽ giúp tái tạo lại các kết quả được trình bày trong bài báo, so sánh thời gian giải trung bình và đường cong phân phối tích lũy thực nghiệm (ECD).

## 📈 Kết quả nổi bật

Phương pháp `nesPDHG` được đề xuất cho thấy sự cải thiện đáng kể về thời gian giải quyết trung bình so với các phương pháp cơ sở.

* **Tại độ chính xác 10⁻⁴ (Figure 1):**
    * `nesPDHG` đạt thời gian giải trung bình thấp nhất (khoảng 22 giây).
    * `nesPDHG` giải được 285 instances, nhiều hơn 3 instances so với `r2HPDHG` (282).

* **Tại độ chính xác 10⁻⁸ (Figure 3):**
    * `nesPDHG` tiếp tục dẫn đầu với thời gian trung bình khoảng 63 giây.
    * `nesPDHG` giải được 268 instances, nhiều hơn 21 instances so với `r2HPDHG` (247).



## 📚 Trích dẫn (Citation)

Nếu bạn sử dụng công trình này trong nghiên cứu của mình, vui lòng trích dẫn bài báo gốc.

```bibtex
@article{NesLP2025,
  title   = {Nesterov–Halpern Methods for LPs},
  author  = {Vu, Thi Huong and Le, Duc Hiep and Koch, Thorsten},
  journal = {ZIB Report (ArXiv Preprint)},
  year    = {2025},
  month   = {November}
}
