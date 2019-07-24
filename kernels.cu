/*
  kernels.cu -- CUDA matrix multiplicaion kernels
  -- as declared in multiply.cuh
  by Cody Rivera
 */

#include "cuda_runtime.h"
#include "multiply.cuh"



__global__ void naiveGEMMKernel(const float* A, const float* B, float* C,
                                const unsigned int m, const unsigned int n,
                                const unsigned int k)
{
    float aVal, bVal, elt;
    unsigned int i = (threadIdx.x + (blockIdx.x * blockDim.x));
    unsigned int j = (threadIdx.y + (blockIdx.y * blockDim.y));
    
    // Loops in case threads are insufficient
    while (i < m)
    {
        while (j < n)
        {
            elt = 0.0;
            for (unsigned int a = 0; a < k; a++)
            {
                aVal = A[i + (a * m)];
                bVal = B[a + (j * k)];
                // Inefficient Accesses
                elt += aVal * bVal;
            }
            C[i + (m * j)] = elt; 
            j += (blockDim.y * gridDim.y);
        }
        i += (blockDim.x * gridDim.x);
    }
}

__global__ void sharedGEMMKernel(const float* A, const float* B, float* C,
                                 const unsigned int m, const unsigned int n,
                                 const unsigned int k)
{
    extern __shared__ float memory[];
    // Simulate 2 arrays
    float* tileA = &memory[0];
    float* tileB = &memory[(blockDim.x * blockDim.y) + blockDim.x];
    
    // Array placement variables
    int iTile = blockIdx.x;
    int iTileMax = (m / blockDim.x) + 1;
    int jTile = blockIdx.y;
    int jTileMax = (m / blockDim.y) + 1;
    
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int j = threadIdx.y + (blockIdx.y * blockDim.y);

    float elt;
    int aMax = k / blockDim.y;
    int iKIndex, jKIndex;
    
    while (iTile < iTileMax)
    {
        while (jTile < jTileMax)
        {
            elt = 0.0;
            // Works on full blocks of A and B
            for (int a = 0; a < aMax; a++)
            {
                // Load shared memory -- All threads participate
                iKIndex = threadIdx.x + (a * blockDim.x);
                jKIndex = threadIdx.y + (a * blockDim.y);
                // Access pattern avoids bank conflicts
                if (i < m)
                {
                    tileA[threadIdx.x
                          + (threadIdx.y * blockDim.x) 
                          + threadIdx.y] = A[i + (m * jKIndex)];
                }
                if (j < m)
                {
                    tileB[threadIdx.x
                          + (threadIdx.y * blockDim.x)
                          + threadIdx.y] = B[iKIndex + (k * j)];
                }
                __syncthreads();
                // Multiplication
                for (int b = 0; b < blockDim.y; b++)
                {
                    elt += tileA[threadIdx.x + (blockDim.x * b) + b]
                        * tileB[b + (blockDim.y * threadIdx.y) + threadIdx.y];
                }
                __syncthreads();
            }
               
            // Handles edge cases, then stores element
            if (i < m && j < n)
            {
                if (aMax * blockDim.y < k)
                {
                    for (int b = (aMax * blockDim.y); b < k; b++)
                    {
                        elt += A[i + (m * b)] * B[b + (k * j)];
                    }
                }
                C[i + (m * j)] = elt; 
            }
            j += (blockDim.y * gridDim.y);
            jTile += gridDim.y;
        }
        i += (blockDim.x * gridDim.x);
        iTile += gridDim.x;
    }
}


// Optimizations inspired by
// https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/
__global__ void optGEMMKernel(const float* A, const float* B, float* C,
                              const unsigned int m, const unsigned int n,
                              const unsigned int k)
{
    extern __shared__ float memory[];
    // Simulate 2 arrays
    float* tileA = &memory[0];
    float* tileB = &memory[(blockDim.x * blockDim.y) + blockDim.x];
    
    // Array placement variables
    int iTile = blockIdx.x;
    int iTileMax = (m / blockDim.x) + 1;
    int jTile = blockIdx.y;
    int jTileMax = (m / blockDim.y) + 1;
    
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int j = threadIdx.y + (blockIdx.y * blockDim.y);

    float elt;
    int aMax = k / blockDim.y;
    int iKIndex, jKIndex;
    
    while (iTile < iTileMax)
    {
        while (jTile < jTileMax)
        {
            elt = 0.0;
            // Works on full blocks of A and B
            for (int a = 0; a < aMax; a++)
            {
                // Load shared memory -- All threads participate
                iKIndex = threadIdx.x + (a * blockDim.x);
                jKIndex = threadIdx.y + (a * blockDim.y);
                // Access pattern avoids bank conflicts
                if (i < m)
                {
                    tileA[threadIdx.x
                          + (threadIdx.y * blockDim.x) 
                          + threadIdx.y] = A[i + (m * jKIndex)];
                }
                if (j < m)
                {
                    tileB[threadIdx.x
                          + (threadIdx.y * blockDim.x)
                          + threadIdx.y] = B[iKIndex + (k * j)];
                }
                __syncthreads();
                // Multiplication
                #pragma unroll
                for (int b = 0; b < TILE_WIDTH; b++)
                {
                    elt += tileA[threadIdx.x + (blockDim.x * b) + b]
                        * tileB[b + (blockDim.y * threadIdx.y) + threadIdx.y];
                }
                __syncthreads();
            }
               
            // Handles edge cases, then stores element
            if (i < m && j < n)
            {
                if (aMax * blockDim.y < k)
                {
                    for (int b = (aMax * blockDim.y); b < k; b++)
                    {
                        elt += A[i + (m * b)] * B[b + (k * j)];
                    }
                }
                C[i + (m * j)] = elt; 
            }
            j += (blockDim.y * gridDim.y);
            jTile += gridDim.y;
        }
        i += (blockDim.x * gridDim.x);
        iTile += gridDim.x;
    }
}

