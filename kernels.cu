/*
  kernels.cu -- TSM2 Kernels -- as declared in multiply.cuh
  by Cody Rivera
 */

#include "cuda_runtime.h"
#include "multiply.cuh"




template <int t1, int t2, int t3>
__global__ void floatTSM2Kernel(const float* A, const float* B, float* C,
                                const unsigned int m, const unsigned int n,
                                const unsigned int k)
{
    // Names mostly follow the published code
    extern __shared__ float currB[];
    
    float currA[t3];
    float nextA[t3];
    float nextB[t2];
    float currC[t2];
        
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    int thread = threadIdx.x + (blockIdx.x * blockDim.x);

    
    for (int p = 0; p < k; p += t2)
    {
        for (int j = 0; j < n; j += t1)
        {
            for (int l = j; l < j + t1; l += t3)
            {
            }
        }
        
    }
    
}

template <int t1, int t2, int t3>
__global__ void doubleTSM2Kernel(const double* A, const double* B, double* C,
                                 const unsigned int m, const unsigned int n,
                                 const unsigned int k);



