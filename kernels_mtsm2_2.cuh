/*
  kernels_mtsm2_2.cuh -- TSM2 Kernels -- as declared in multiply.cuh

  Second round multiple TSM2 optimization

  by Cody Rivera
*/

#include "cuda_runtime.h"
#include "multiply.cuh"




// NOTE- m is the common dimension between matrix A and B
template <int t1, int t2, int t3>
__global__ void floatTSM2Kernel(const float* A, const float* B, float* C,
                                const unsigned int m, const unsigned int n,
                                const unsigned int k)
{
    // Names mostly follow the published code
    __shared__ float currB[t1 * t2];
    
    float currA[t3];
    float nextA[t3];
    float nextB[t2];
    float currC[t2];
    float nextC[t2];
        
    const int tid = threadIdx.x;
    int threadBase = (blockIdx.x * blockDim.x);
    int thread = threadBase + tid;

    // Initial load - This'll have to be done anyway, as we're scheduling the loads to happen concurrently with computation
    // Loads first tile of output registers and A
    if (thread < m)
    {
        #pragma unroll
        for (int i = 0; i < t2; ++i)
        {
            if (i < n)
            {
                currC[i] = C[thread + (i * m)];
            }
        }
        // Loads currA
        #pragma unroll
        for (int i = 0; i < t3; ++i)
        {
            if (i < k)
            {
                currA[i] = A[thread + (i * m)];
            }
        }
    }
    // Loads tile of B
    if (tid < k)
    {
        #pragma unroll
        for (int i = 0; i < t2; ++i)
        {
            if (i < n)
            {
                currB[tid + (i * t1)] = B[tid + (i * k)];
            }
        }
    }

    // We cannot rule out a thread's participation based on
    // whether it corresponds to a row in Matrix A, so we
    // introduce threadBase.
    for (; threadBase < m; threadBase += blockDim.x * gridDim.x)
    {
        thread = threadBase + tid;
        for (int p = 0; p < n; p += t2)
        {
            bool loadNewC = false;
            // Outer product loop
            for (int j = 0; j < k; j += t1)
            {
                __syncthreads();
                // If there are more rows of Matrix B
                if (k > j + t1)
                {
                    #pragma unroll
                    for (int i = 0; i < t2; ++i)
                    {
                        if (j + t1 + tid < k && p + i < n)
                        {
                            nextB[i] = B[(j + t1 + tid) + ((p + i) * k)]; 
                        }
                    }
                }
                // If there are more columns of B to consider
                else if (n > p + t2)
                {
                    #pragma unroll
                    for (int i = 0; i < t2; ++i)
                    {
                        if (tid < k && p + t2 + i < n)
                        {
                            nextB[i] = B[tid + ((p + t2 + i) * k)]; 
                        }
                    }

                    loadNewC = true;
                    #pragma unroll
                    for (int i = 0; i < t2; ++i)
                    {
                        if (p + i + t2 < n && thread < m)
                        {
                            nextC[i] = C[thread + ((p + t2 + i) * m)];
                        }
                    }
                }
                // If there is another tc-row tile of A to consider
                else if (m > threadBase + (blockDim.x * gridDim.x))
                {
                    #pragma unroll
                    for (int i = 0; i < t2; ++i)
                    {
                        if (tid < k && i < n)
                        {
                            nextB[i] = B[tid + (i * k)]; 
                        }
                    }

                    loadNewC = true;
                    #pragma unroll
                    for (int i = 0; i < t2; ++i)
                    {
                        if (i < n && thread + (blockDim.x * gridDim.x) < m)
                        {
                            nextC[i] = C[thread + (blockDim.x * gridDim.x) + (i * m)];
                        }
                    }
                }
                
                const int t3mod = t1 % t3;
                
                // Loop over A's columns 
                for (int l = j; l < j + (t1 - t3mod) && l < k; l += t3)
                {
                    bool lastIter = false;
                    // Loads next A
                    if (j + (t1 - t3mod) > l + t3 && k > l + t3)
                    {
                        #pragma unroll
                        for (int i = 0; i < t3; ++i)
                        {
                            if (l + t3 + i < k && thread < m)
                            {
                                nextA[i] = A[thread + ((l + t3 + i) * m)];
                            }
                        }
                    }
                    // Loads start of next tile of A that corresponds with next tile of B
                    else if (k > j + t1)
                    {
                        lastIter = true;
                        #pragma unroll
                        for (int i = 0; i < t3; ++i)
                        {
                            if (j + t1 + i < k && thread < m)
                            {
                                nextA[i] = A[thread + ((j + t1 + i) * m)];
                            }
                        }
                    }
                    // If there are more columns of B to consider
                    else if (n > p + t2)
                    {
                        #pragma unroll
                        for (int i = 0; i < t3; ++i)
                        {
                            if (i < k && thread < m)
                            {
                                nextA[i] = A[thread + (i * m)];
                            }
                        }
                    }
                    // If theres is another tile of A to consider
                    else if (m > threadBase + (blockDim.x * gridDim.x))
                    {
                        #pragma unroll
                        for (int i = 0; i < t3; ++i)
                        {
                            if (i < k && thread + (blockDim.x * gridDim.x) < m)
                            {
                                nextA[i] = A[thread + (blockDim.x * gridDim.x) + (i * m)];
                            }
                        }
                    }
                     
                    // Floating Point Operations (lines 32-34)
                    // Each thread does t2 * t3 mults
                   
                    // Either dispatch guarded or unguarded instructions based on 
                    // position in matrix A
                    if (l + t3 <= k)
                    {
                        // It is assumed that B[(l - j) .. (l - j) + t3 - 1, _]  exist
                        #pragma unroll
                        for (int a = 0; a < t2; ++a)
                        {
                            #pragma unroll
                            for (int b = 0; b < t3; ++b)
                            {
                                currC[a] += currA[b] * currB[(l - j) + b + (a * t1)]; 
                            }
                        }
                    }
                    else
                    {
                        #pragma unroll
                        for (int a = 0; a < t2; ++a)
                        {
                            #pragma unroll
                            for (int b = 0; b < t3; ++b)
                            {
                                if (l + b < k)
                                {
                                    currC[a] += currA[b] * currB[(l - j) + b + (a * t1)]; 
                                }
                            }
                        }
                    }                   

                    if (lastIter)
                    {
                        // Accommodates t3 that do not divide t1.
                        #pragma unroll
                        for (int a = 0; a < t2; ++a)
                        {
                            #pragma unroll
                            for (int b = 0; b < t3mod; ++b)
                            {
                                if (j + t1 - t3mod + b < k)
                                {
                                    currC[a] += currA[b] * currB[(t1 - t3mod + b) + (a * t1)];
                                }
                            }
                        }
                    }

                    // Stores next A in curr A
                    #pragma unroll
                    for (int i = 0; i < t3; ++i)
                    {
                        currA[i] = nextA[i];
                    }
                }

                __syncthreads();

                // Loads currB from each thread's nextB
                #pragma unroll
                for (int i = 0; i < t2; ++i)
                {
                    currB[tid + (i * t1)] = nextB[i];
                }
                
            }
            // Stores C
            if (thread < m)
            {
                #pragma unroll
                for (int i = 0; i < t2; ++i)
                {
                    if (p + i < n)
                    {
                        C[thread + ((p + i) * m)] = currC[i];
                    }
                }
            }
            // If applicable, loads nextC to currC
            if (loadNewC)
            {
                #pragma unroll
                for (int i = 0; i < t2; ++i)
                {
                    currC[i] = nextC[i];
                }
            }
        }
    }    
}


template <int t1, int t2, int t3>
__global__ void doubleTSM2Kernel(const double* A, const double* B, double* C,
                                 const unsigned int m, const unsigned int n,
                                 const unsigned int k)
{
    // Names mostly follow the published code
    __shared__ double currB[t1 * t2];
    
    double currA[t3];
    double nextA[t3];
    double nextB[t2];
    double currC[t2];
    double nextC[t2];


    const int tid = threadIdx.x;
    int threadBase = (blockIdx.x * blockDim.x);
    int thread = threadBase + tid;

    // Initial load - This'll have to be done anyway, as we're scheduling the loads to happen concurrently with computation
    // Loads first tile of output registers and A
    if (thread < m)
    {
        #pragma unroll
        for (int i = 0; i < t2; ++i)
        {
            if (i < n)
            {
                currC[i] = C[thread + (i * m)];
            }
        }
        // Loads currA
        #pragma unroll
        for (int i = 0; i < t3; ++i)
        {
            if (i < k)
            {
                currA[i] = A[thread + (i * m)];
            }
        }
    }
    // Loads tile of B
    if (tid < k)
    {
        #pragma unroll
        for (int i = 0; i < t2; ++i)
        {
            if (i < n)
            {
                currB[tid + (i * t1)] = B[tid + (i * k)];
            }
        }
    }

    // We cannot rule out a thread's participation based on
    // whether it corresponds to a row in Matrix A, so we
    // introduce threadBase.
    for (; threadBase < m; threadBase += blockDim.x * gridDim.x)
    {
        thread = threadBase + tid;
        for (int p = 0; p < n; p += t2)
        {
            bool loadNewC = false;
            // Outer product loop
            for (int j = 0; j < k; j += t1)
            {
                __syncthreads();
                // If there are more rows of Matrix B
                if (k > j + t1)
                {
                    #pragma unroll
                    for (int i = 0; i < t2; ++i)
                    {
                        if (j + t1 + tid < k && p + i < n)
                        {
                            nextB[i] = B[(j + t1 + tid) + ((p + i) * k)]; 
                        }
                    }
                }
                // If there are more columns of B to consider
                else if (n > p + t2)
                {
                    #pragma unroll
                    for (int i = 0; i < t2; ++i)
                    {
                        if (tid < k && p + t2 + i < n)
                        {
                            nextB[i] = B[tid + ((p + t2 + i) * k)]; 
                        }
                    }

                    loadNewC = true;
                    #pragma unroll
                    for (int i = 0; i < t2; ++i)
                    {
                        if (p + i + t2 < n && thread < m)
                        {
                            nextC[i] = C[thread + ((p + t2 + i) * m)];
                        }
                    }
                }
                // If there is another tc-row tile of A to consider
                else if (m > threadBase + (blockDim.x * gridDim.x))
                {
                    #pragma unroll
                    for (int i = 0; i < t2; ++i)
                    {
                        if (tid < k && i < n)
                        {
                            nextB[i] = B[tid + (i * k)]; 
                        }
                    }

                    loadNewC = true;
                    #pragma unroll
                    for (int i = 0; i < t2; ++i)
                    {
                        if (i < n && thread + (blockDim.x * gridDim.x) < m)
                        {
                            nextC[i] = C[thread + (blockDim.x * gridDim.x) + (i * m)];
                        }
                    }
                }
                
                const int t3mod = t1 % t3;
                
                // Loop over A's columns 
                for (int l = j; l < j + (t1 - t3mod) && l < k; l += t3)
                {
                    bool lastIter = false;
                    // Loads next A
                    if (j + (t1 - t3mod) > l + t3 && k > l + t3)
                    {
                        #pragma unroll
                        for (int i = 0; i < t3; ++i)
                        {
                            if (l + t3 + i < k && thread < m)
                            {
                                nextA[i] = A[thread + ((l + t3 + i) * m)];
                            }
                        }
                    }
                    // Loads start of next tile of A that corresponds with next tile of B
                    else if (k > j + t1)
                    {
                        lastIter = true;
                        #pragma unroll
                        for (int i = 0; i < t3; ++i)
                        {
                            if (j + t1 + i < k && thread < m)
                            {
                                nextA[i] = A[thread + ((j + t1 + i) * m)];
                            }
                        }
                    }
                    // If there are more columns of B to consider
                    else if (n > p + t2)
                    {
                        #pragma unroll
                        for (int i = 0; i < t3; ++i)
                        {
                            if (i < k && thread < m)
                            {
                                nextA[i] = A[thread + (i * m)];
                            }
                        }
                    }
                    // If theres is another tile of A to consider
                    else if (m > threadBase + (blockDim.x * gridDim.x))
                    {
                        #pragma unroll
                        for (int i = 0; i < t3; ++i)
                        {
                            if (i < k && thread + (blockDim.x * gridDim.x) < m)
                            {
                                nextA[i] = A[thread + (blockDim.x * gridDim.x) + (i * m)];
                            }
                        }
                    }
                     
                    // Floating Point Operations (lines 32-34)
                    // Each thread does t2 * t3 mults
                   
                    // Either dispatch guarded or unguarded instructions based on 
                    // position in matrix A
                    if (l + t3 <= k)
                    {
                        // It is assumed that B[(l - j) .. (l - j) + t3 - 1, _]  exist
                        #pragma unroll
                        for (int a = 0; a < t2; ++a)
                        {
                            #pragma unroll
                            for (int b = 0; b < t3; ++b)
                            {
                                currC[a] += currA[b] * currB[(l - j) + b + (a * t1)]; 
                            }
                        }
                    }
                    else
                    {
                        #pragma unroll
                        for (int a = 0; a < t2; ++a)
                        {
                            #pragma unroll
                            for (int b = 0; b < t3; ++b)
                            {
                                if (l + b < k)
                                {
                                    currC[a] += currA[b] * currB[(l - j) + b + (a * t1)]; 
                                }
                            }
                        }
                    }                   

                    if (lastIter)
                    {
                        // Accommodates t3 that do not divide t1.
                        #pragma unroll
                        for (int a = 0; a < t2; ++a)
                        {
                            #pragma unroll
                            for (int b = 0; b < t3mod; ++b)
                            {
                                if (j + t1 - t3mod + b < k)
                                {
                                    currC[a] += currA[b] * currB[(t1 - t3mod + b) + (a * t1)];
                                }
                            }
                        }
                    }

                    // Stores next A in curr A
                    #pragma unroll
                    for (int i = 0; i < t3; ++i)
                    {
                        currA[i] = nextA[i];
                    }
                }

                __syncthreads();

                // Loads currB from each thread's nextB
                #pragma unroll
                for (int i = 0; i < t2; ++i)
                {
                    currB[tid + (i * t1)] = nextB[i];
                }
                
            }
            // Stores C
            if (thread < m)
            {
                #pragma unroll
                for (int i = 0; i < t2; ++i)
                {
                    if (p + i < n)
                    {
                        C[thread + ((p + i) * m)] = currC[i];
                    }
                }
            }
            // If applicable, loads nextC to currC
            if (loadNewC)
            {
                #pragma unroll
                for (int i = 0; i < t2; ++i)
                {
                    currC[i] = nextC[i];
                }
            }
        }
    }
}

