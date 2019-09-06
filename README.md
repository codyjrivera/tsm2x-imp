
Tall-And-Skinny Matrix Multiplication for CUDA
==============================================

by
Cody Rivera [cjrivera1@crimson.ua.edu],
Jieyang Chen [chenj3@ornl.gov] (First Author), and
Dingwen Tao [tao@cs.ua.edu]

This repository contains an implementation of TSM2 as described by
Chen et al. [1]. TSM2 is a parallel matrix-matrix multiplication algorithm 
optimized for tall and skinny matrices: matrices of size (n * n) and (n * k)
where k is much smaller than n. According to experimental data, this algorithm
is faster and utilizes more memory bandwidth than CUBLAS when multiplying tall
and skinny matrices.

We have implemented the kernels as templates, with the parameters t1, t2, and t3 as
template variables [1]. The program will select an optimal kernel depending on the 
size of the input matrices. Currently, this implementation is only optimized
for the Nvidia V100 GPU.

The implementation also accepts matrices of size (m * n) and (n * k), where m != n.

Please feel free to contact one of us if you have any questions.


Instructions:
-------------

This implementation is designed for Unix platforms, and can be built using
'make'. The usage of this program is: 
'./multiply [-d] a.mtx b.mtx c.mtx',
where a.mtx and b.mtx are input matrices and c.mtx is an output matrix.
Note that the optional parameter [-d] indicates that the matrices are 
double precision.

The format of the matrices is binary, with a structure as follows:

```C++
template <typename FloatType>
struct matrixFormat
{
    uint32 rows, cols;
    FloatType values[rows * cols];
};
```

The matrix is stored in column-major format.
All multibyte values are little-endian.



Sources:
--------

[1] Jieyang Chen, Nan Xiong, Xin Liang, Dingwen Tao, Sihuan Li, Kaiming
Ouyang, Kai Zhao, Nathan DeBardeleben, Qiang Guan, Zizhong Chen.
2019. TSM2: Optimizing Tall-and-Skinny Matrix-Matrix Multiplication on
GPUs. In 2019 International Conference on Supercomputing (ICS ’19), June
26–28, 2019, Phoenix, AZ, USA. ACM, New York, NY, USA, 11 pages. [https:
//doi.org/10.1145/3330345.3330355](https://doi.org/10.1145/3330345.3330355)



