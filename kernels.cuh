/*
  kernels.cu -- TSM2 Kernels -- as declared in multiply.cuh
  by Cody Rivera
 */

#include "cuda_runtime.h"
#include "multiply.cuh"




// NOTE- m is the common dimension between matrix A and B
template <int t1, int t2, int t3>
__global__ void floatTSM2Kernel(const float* A, const float* B, float* C,
                                const unsigned int n, const unsigned int m,
                                const unsigned int k)
{
    // Names mostly follow the published code
    __shared__ float currB[t1 * t2];
    
    float currA[t3];
    float nextA[t3];
    float nextB[t2];
    float currC[t2];
        
    const int tid = threadIdx.x;
    int threadBase = (blockIdx.x * blockDim.x);
    int thread;
    
    // This implementation can respond to arbitrary input

    // We cannot rule out a thread's participation based on
    // whether it corresponds to a row in Matrix A, so we
    // introduce threadBase.
    for (; threadBase < n; threadBase += blockDim.x * gridDim.x)
    {
        thread = threadBase + tid;
        for (int p = 0; p < k; p += t2)
        {
            // Load loops have extra conditionals to ensure
            // they do not make bad memory accesses
            
            // Loads first tile of output registers and A
            if (thread < n)
            {
                #pragma unroll
                for (int i = 0; i < t2; ++i)
                {
                    if (p + i < k)
                    {
                        currC[i] = C[thread + ((p + i) * n)];
                    }
                }
                // Loads currA
                #pragma unroll
                for (int i = 0; i < t3; ++i)
                {
                    if (i < m)
                    {
                        currA[i] = A[thread + (i * n)];
                    }
                }
            }
            // Loads tile of B
            if (tid < m)
            {
                #pragma unroll
                for (int i = 0; i < t2; ++i)
                {
                    if (p + i < k)
                    {
                        currB[tid + (i * t1)] = B[tid + ((p + i) * m)];
                    }
                }
            }

            // Outer product loop
            for (int j = 0; j < m; j += t1)
            {
                __syncthreads();
                // Loads next tile of B
                if (j + t1 + tid < m)
                {
                    #pragma unroll
                    for (int i = 0; i < t2; ++i)
                    {
                        if (p + i < k)
                        {
                            nextB[i] = B[(j + t1 + tid) + ((p + i) * m)]; 
                        }
                    }
                }
                
                const int t3mod = t1 % t3;
                // Two cases - ordinary and edge
                if (m > t3)
                {
                    // Loop over A's columns 
                    for (int l = j; l < j + (t1 - t3mod) && l < m; l += t3)
                    {
                        // Loads next A
                        #pragma unroll
                        for (int i = 0; i < t3; ++i)
                        {
                            if (l + t3 + i < m && thread < n)
                            {
                                nextA[i] = A[thread + ((l + t3 + i) * n)];
                            }
                        }
                                        
                        // Floating Point Operations (lines 32-34)
                        // Each thread does t2 * t3 mults
                        #pragma unroll
                        for (int i = 0; i < t2; ++i)
                        {
                            #pragma unroll
                            for (int k = 0; k < t3; ++k)
                            {
                                currC[i] += currA[k] * currB[(l - j) + k + (i * t1)]; 
                            }
                        }
                    
                        // Stores next A in curr A
                        #pragma unroll
                        for (int i = 0; i < t3; ++i)
                        {
                            currA[i] = nextA[i];
                        }
                    }
                    // Accommodates t3 that do not divide t1.
                    #pragma unroll
                    for (int i = 0; i < t2; ++i)
                    {
                        #pragma unroll
                        for (int k = 0; k < t3mod; ++k)
                        {
                            if (j + t1 - t3mod + k < m)
                            {
                                currC[i] += currA[k] * currB[(t1 - t3mod + k) + (i * t1)];
                            }
                        }
                    }
                }
                else
                {
                    #pragma unroll
                    for (int i = 0; i < t2; ++i)
                    {
                        #pragma unroll
                        for (int k = 0; k < t3; ++k)
                        {
                            if (j + k < m)
                            {
                                currC[i] += currA[k] * currB[k + (i * t1)];
                            }
                        }
                    }
                }

                __syncthreads();

                // Loads currB from each thread's nextB
                #pragma unroll
                for (int i = 0; i < t2; ++i)
                {
                    currB[tid + (i * t1)] = nextB[i];
                }
                
                // Loads next currA
                if (t3mod != 0)
                {
                    #pragma unroll
                    for (int i = 0; i < t3; ++i)
                    {
                        if (j + t1 + i < m && thread < n)
                        {
                            currA[i] = A[thread + ((j + t1 + i) * n)];
                        }
                    }
                }
            }
            // Stores C
            if (thread < n)
            {
                #pragma unroll
                for (int i = 0; i < t2; ++i)
                {
                    if (p + i < k)
                    {
                        C[thread + ((p + i) * n)] = currC[i];
                    }
                }
            }
        }
    }    
}

template <int t1, int t2, int t3>
__global__ void doubleTSM2Kernel(const double* A, const double* B, double* C,
                                 const unsigned int n, const unsigned int m,
                                 const unsigned int k)
{
    // Names mostly follow the published code
    __shared__ float currB[t1 * t2];
    
    float currA[t3];
    float nextA[t3];
    float nextB[t2];
    float currC[t2];
        
    const int tid = threadIdx.x;
    int threadBase = (blockIdx.x * blockDim.x);
    int thread;
    
    // This implementation can respond to arbitrary input

    // We cannot rule out a thread's participation based on
    // whether it corresponds to a row in Matrix A, so we
    // introduce threadBase.
    for (; threadBase < n; threadBase += blockDim.x * gridDim.x)
    {
        thread = threadBase + tid;
        for (int p = 0; p < k; p += t2)
        {
            // Load loops have extra conditionals to ensure
            // they do not make bad memory accesses
            
            // Loads first tile of output registers and A
            if (thread < n)
            {
                #pragma unroll
                for (int i = 0; i < t2; ++i)
                {
                    if (p + i < k)
                    {
                        currC[i] = C[thread + ((p + i) * n)];
                    }
                }
                // Loads currA
                #pragma unroll
                for (int i = 0; i < t3; ++i)
                {
                    if (i < m)
                    {
                        currA[i] = A[thread + (i * n)];
                    }
                }
            }
            // Loads tile of B
            if (tid < m)
            {
                #pragma unroll
                for (int i = 0; i < t2; ++i)
                {
                    if (p + i < k)
                    {
                        currB[tid + (i * t1)] = B[tid + ((p + i) * m)];
                    }
                }
            }

            // Outer product loop
            for (int j = 0; j < m; j += t1)
            {
                __syncthreads();
                // Loads next tile of B
                if (j + t1 + tid < m)
                {
                    #pragma unroll
                    for (int i = 0; i < t2; ++i)
                    {
                        if (p + i < k)
                        {
                            nextB[i] = B[(j + t1 + tid) + ((p + i) * m)]; 
                        }
                    }
                }
                
                const int t3mod = t1 % t3;
                // Two cases - ordinary and edge
                if (m > t3)
                {
                    // Loop over A's columns 
                    for (int l = j; l < j + (t1 - t3mod) && l < m; l += t3)
                    {
                        // Loads next A
                        #pragma unroll
                        for (int i = 0; i < t3; ++i)
                        {
                            if (l + t3 + i < m && thread < n)
                            {
                                nextA[i] = A[thread + ((l + t3 + i) * n)];
                            }
                        }
                                        
                        // Floating Point Operations (lines 32-34)
                        // Each thread does t2 * t3 mults
                        #pragma unroll
                        for (int i = 0; i < t2; ++i)
                        {
                            #pragma unroll
                            for (int k = 0; k < t3; ++k)
                            {
                                currC[i] += currA[k] * currB[(l - j) + k + (i * t1)]; 
                            }
                        }
                    
                        // Stores next A in curr A
                        #pragma unroll
                        for (int i = 0; i < t3; ++i)
                        {
                            currA[i] = nextA[i];
                        }
                    }
                    // Accommodates t3 that do not divide t1.
                    #pragma unroll
                    for (int i = 0; i < t2; ++i)
                    {
                        #pragma unroll
                        for (int k = 0; k < t3mod; ++k)
                        {
                            if (j + t1 - t3mod + k < m)
                            {
                                currC[i] += currA[k] * currB[(t1 - t3mod + k) + (i * t1)];
                            }
                        }
                    }
                }
                else
                {
                    #pragma unroll
                    for (int i = 0; i < t2; ++i)
                    {
                        #pragma unroll
                        for (int k = 0; k < t3; ++k)
                        {
                            if (j + k < m)
                            {
                                currC[i] += currA[k] * currB[k + (i * t1)];
                            }
                        }
                    }
                }

                __syncthreads();

                // Loads currB from each thread's nextB
                #pragma unroll
                for (int i = 0; i < t2; ++i)
                {
                    currB[tid + (i * t1)] = nextB[i];
                }
                
                // Loads next currA
                if (t3mod != 0)
                {
                    #pragma unroll
                    for (int i = 0; i < t3; ++i)
                    {
                        if (j + t1 + i < m && thread < n)
                        {
                            currA[i] = A[thread + ((j + t1 + i) * n)];
                        }
                    }
                }
            }
            // Stores C
            if (thread < n)
            {
                #pragma unroll
                for (int i = 0; i < t2; ++i)
                {
                    if (p + i < k)
                    {
                        C[thread + ((p + i) * n)] = currC[i];
                    }
                }
            }
        }
    }
}




