
TSM2: Tall-And-Skinny Matrix Multiplication for CUDA
====================================================

by
Cody Rivera [cjrivera1@crimson.ua.edu],
Jieyang Chen [chenj3@ornl.gov] (First Author), and
Dingwen Tao [tao@cs.ua.edu] (Supervisor)

This repository contains an implementation of TSM2 as described by
Chen et al. [1]. TSM2 is a parallel matrix-matrix multiplication algorithm 
optimized for tall and skinny matrices: matrices of size (m * m) and (m * n)
where n is much smaller than m [2]. According to experimental data, this algorithm
is faster and utilizes more memory bandwidth than CUBLAS when multiplying tall
and skinny matrices.

We have implemented the kernels as templates, with the parameters t1, t2, and t3 as
template variables [1]. The program will select an optimal kernel depending on the 
size of the input matrices. Currently, this implementation is only optimized
for the Nvidia V100 GPU.

The implementation also accepts matrices of size (m * k) and (k * n), where m != k.

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

You may use the provided gen.cpp program to generate input
matrices. The usage is ./gen [-d] -r ROW_COUNT -c COL_COUNT file,
where -d signifies double precision.


Notes:
------

[1] Chen, Jieyang, Nan Xiong, Xin Liang, Dingwen Tao, Sihuan Li, Kaiming Ouyang, Kai Zhao, Nathan DeBardeleben, Qiang Guan, and Zizhong Chen. 
"TSM2: optimizing tall-and-skinny matrix-matrix multiplication on GPUs." 
In Proceedings of the ACM International Conference on Supercomputing (ICS), pp. 106-116. ACM, 2019. 
[https://doi.org/10.1145/3330345.3330355](https://doi.org/10.1145/3330345.3330355)

[2] In [1], the matrices are presented as being of size (n * n) and (n * k). We have relabeled our matrices to be more consistent
with CUBLAS.



