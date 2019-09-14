#ifndef _MULTIPLY_CUH
#define _MULTIPLY_CUH

/*
  multiply.cuh -- Matrix multiplication interface and useful
  macros - by Cody Rivera
 */

#include <cstdio>
#include "cuda_runtime.h"
#include "cublas_v2.h"

// Useful macros
#define cudaErrchk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"CUDAassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define cublasErrchk(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
    if (code != CUBLAS_STATUS_SUCCESS) 
    {
        fprintf(stderr,"CUBLASassert Failure Code: %d %s %d\n", code, file, line);
        if (abort) exit(code);
    }
}



// Functions
template<typename FloatType>
bool runKernels(const FloatType* A, const FloatType* B, FloatType* C,
                const unsigned int m, const unsigned int n,
                const unsigned int k);


/* 
   TSM Matrix Multiplication Interface

   Inputs - passed const:
   m, n, k - Unsigned integer dimensions
   A - Column major matrix m * k with leading dimension m
   B - Column major matrix k * n with leading dimension k
   
   Outputs:
   C - Column major matrix m * n with leading dimension m
 */


/*

template <int t1, int t2, int t3>
__global__ void floatTSM2Kernel(const float* A, const float* B, float* C,
                                const unsigned int m, const unsigned int n,
                                const unsigned int k);

template <int t1, int t2, int t3>
__global__ void doubleTSM2Kernel(const double* A, const double* B, double* C,
                                 const unsigned int m, const unsigned int n,
                                 const unsigned int k);

*/
#endif