# NesPDHG: Halpern-typed Methods for Linear Programming

This is the code repository for the research paper: **"Halpern-typed methods for LPs"**.

This work investigates Halpern-typed methods for solving linear programs (LPs). The code in this repository includes implementations of the described algorithms, as well as numerical experiments to compare them against other state-of-the-art first-order methods.

This code is built upon the **MPAX** (Mathematical Programming in JAX) library (original: `https://github.com/MIT-Lu-Lab/MPAX`).

## 📄 Related Paper (Preparing)

* **Paper Title:** Nesterov–Halpern Methods for LPs
* **Authors:** Vu Thi Huong, **Le Duc Hiep**, and Thorsten Koch

> **Abstract:** In this work, we study Halpern-typed methods to solve linear programs. Theoretical guarantees for the convergence and convergence rates of the methods are revised, and numerical experiments to compare with state-of-the-art first-order methods are presented.

## 🚀 Implemented Algorithms

This repository extends the original `MPAX` library with the following algorithms:

* **`nesPDHG`**: The proposed Halpern-typed method from this work, based on its connection to Nesterov's acceleration. In experiments, it is configured with `w=3` and `gamma=0.75`.
* **`nes1_pdhg`**, **`nes2_pdhg`**: Variants of `nesPDHG` with different parameter choices for `w` and `gamma`.
* **`r2HPDHG`**: A reflection "Restarted Halpern PDHG" variant.
* **`rHPDHG`**: The baseline "Restarted Halpern PDHG" method.
* **`r2HPDHGmpax`**: The practical implementation of `r2HPDHG` found in the `MPAX` library.

## 🛠️ Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/hiepday3324/NesPDHG.git](https://github.com/hiepday3324/NesPDHG.git)
    cd NesPDHG
    ```

2.  Install dependencies. This project uses JAX and was tested on an NVIDIA RTX 4090 GPU.
    ```bash
    # Install dependencies from requirements.txt (if available)
    pip install -r requirements.txt
    
    # Or manually install the main libraries
    pip install jax jaxlib numpy pandas
    ```

## 📊 Reproducing Results

This notebook will help reproduce the results presented in the paper, comparing average solve times and empirical cumulative distribution (ECD) curves.

## 📈 Key Results

The proposed `nesPDHG` method shows a significant improvement in average solve time compared to baseline methods.

* **At 10⁻⁴ Accuracy (Figure 1):**
    * `nesPDHG` achieves the lowest average solve time (approx. 22 seconds).
    * `nesPDHG` solves 285 instances, 3 more than `r2HPDHG` (282).

* **At 10⁻⁸ Accuracy (Figure 3):**
    * `nesPDHG` continues to lead with an average time of approx. 63 seconds.
    * `nesPDHG` solves 268 instances, 21 more than `r2HPDHG` (247).

## 📚 Citation

If you use this work in your research, please cite the original paper.

```bibtex
@article{NesLP2025,
  title   = {Nesterov–Halpern Methods for LPs},
  author  = {Vu, Thi Huong and Le, Duc Hiep and Koch, Thorsten},
  journal = {ZIB Report (ArXiv Preprint)},
  year    = {2025},
  month   = {November}
}
