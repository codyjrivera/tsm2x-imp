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
bool runKernels(const float* A, const float* B, float* C,
                const unsigned int m, const unsigned int n,
                const unsigned int k);


/* 
   Matrix Multiplication Interface

   Inputs - passed const:
   m, n, k - Unsigned integer dimensions
   A - Column major matrix m * k with leading dimension k
   B - Column major matrix k * n with leading dimension n
   
   Outputs:
   C - Column major matrix m * n with leading dimension n
 */


/*
  Implementations - naive O(n^3) multiplication
 */

__global__ void naiveGEMMKernel(const float* A, const float* B, float* C,
                                const unsigned int m, const unsigned int n, 
                                const unsigned int k);

__global__ void sharedGEMMKernel(const float* A, const float* B, float* C,
                                 const unsigned int m, const unsigned int n, 
                                 const unsigned int k);

__global__ void optGEMMKernel(const float* A, const float* B, float* C,
                              const unsigned int m, const unsigned int n, 
                              const unsigned int k);


// Implementation macros
#define TILE_WIDTH 16
#define TTILE_WIDTH 8
