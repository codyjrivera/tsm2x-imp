
TSM2X: High-Performance Tall-and-Skinny Matrix-Matrix Multiplication on GPUs
=============================================================

by
Cody Rivera [cjrivera1@crimson.ua.edu],
Jieyang Chen [chenj3@ornl.gov], and
Dingwen Tao [dingwen.tao@wsu.edu]

This repository contains an implementation of two irregular-shape matrix-matrix
multiplication algorithms, `TSM2R` and `TSM2L`. `TSM2R` is designed to efficiently
multiply a large square (or near-square) matrix by a tall-and-skinny matrix, or
more specifically, an (m * k) and (k * n) matrix-matrix multiplication where
m and k are approximately equal, and n is much smaller than k. `TSM2L` is designed
to efficiently multiply a tall-and-skinny matrix by a small square matrix, or
more specifically, an (m * k) and (k * n) matrix-matrix multiplication where 
k is much smaller than m, and k and n are approximately equal.

We propose `TSM2R` and `TSM2L` in our preprint,
"TSM2X: High-Performance Tall-and-Skinny Matrix-MatrixMultiplication on GPUs." [1].
Our work extends an ICS conference paper [2], which introduces `TSM2R`, by expanding
its techniques for different matrix sizes as well as porting the algorithm to the Nvidia
Tesla V100.

We have implemented the kernels as templates, with the parameters `t1`, `t2`, and `t3` as
template variables [1]. The program will select an optimal kernel depending on the 
size of the input matrices. This repository currently provides a set of optimal kernels for
the Nvidia V100 GPU only.

Instructions:
-------------

This implementation is designed for Unix platforms, and can be built using
`make`. The usage of this program is: 
`./multiply [-d] [-i] a.mtx b.mtx c.mtx`,
where a.mtx and b.mtx are input matrices and c.mtx is an output matrix.
`-d` indicates that the matrices are double-precision, while `-i` indicates
that `ISM2` is to be used.


The format of the matrices is binary, with a structure as follows:

```C++
template <typename FloatType>
struct matrixFormat {
    uint32_t rows, cols;
    FloatType values[rows * cols];
};
```

The matrix is stored in column-major format.
All multibyte values are little-endian.

You may use the provided gen.cpp program to generate input
matrices. The usage is `./gen [-d] -r ROW_COUNT -c COL_COUNT file`,
where `-d` signifies double precision.

You may also use the provided print.cpp program to print matrices.
The usage is `./print [-d] file`.

To evaluate performance across a range of inputs, a Python3 script
`test.py` is provided. The script can be invoked with 
`python3 test.py`. The program requires that `../multiply` and
`../gen` exist, and writes its output to CSV files.

Notes:
------

[1] Cody Rivera, Jieyang Chen, Nan Xiong, Shuaiwen Leon Song, and Dingwen Tao. "TSM2X: High-Performance Tall-and-Skinny Matrix-MatrixMultiplication on GPUs." 
2020. [arXiv:2002.03258](https://arxiv.org/abs/2002.03258) [cs.DC].

[2] Jieyang Chen, Nan Xiong, Xin Liang, Dingwen Tao, Sihuan Li, Kaiming Ouyang, Kai Zhao, Nathan DeBardeleben, Qiang Guan, and Zizhong Chen. 
"TSM2: optimizing tall-and-skinny matrix-matrix multiplication on GPUs." 
In Proceedings of the ACM International Conference on Supercomputing (ICS), pp. 106-116. ACM, 2019. 
[https://doi.org/10.1145/3330345.3330355](https://doi.org/10.1145/3330345.3330355)
