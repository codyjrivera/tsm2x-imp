/**
 * Original TSM2 Kernel. This follows the original paper's optimized
 * pseudocode, with extra code to handle edge cases.
 *
 * by Cody Rivera, 2019-2020
 */

#ifndef _KERNEL_TSM2_CUH
#define _KERNEL_TSM2_CUH

#include "cuda_runtime.h"
 
template <typename FloatType, int t1, int t2, int t3>
__global__ void kernelTsm2(const FloatType* A, const FloatType* B, FloatType* C,
                        const unsigned int m, const unsigned int n,
                        const unsigned int k) {
    // Names mostly follow the paper's
    __shared__ FloatType currB[t1 * t2];

    FloatType currA[t3];
    FloatType nextA[t3];
    FloatType nextB[t2];
    FloatType currC[t2];

    const int tid = threadIdx.x;
    int threadBase = (blockIdx.x * blockDim.x);
    int thread;

    // This implementation can respond to arbitrary input

    // We cannot rule out a thread's participation based on
    // whether it corresponds to a row in Matrix A, so we
    // introduce threadBase.
    for (; threadBase < m; threadBase += blockDim.x * gridDim.x) {
        thread = threadBase + tid;
        for (int p = 0; p < n; p += t2) {
            // Load loops have extra conditionals to ensure
            // they do not make bad memory accesses

            // Loads first tile of output registers and A
            if (thread < m) {
                #pragma unroll
                for (int i = 0; i < t2; ++i) {
                    if (p + i < n) {
                        currC[i] = C[thread + ((p + i) * m)];
                    }
                }
                // Loads currA
                #pragma unroll
                for (int i = 0; i < t3; ++i) {
                    if (i < k) {
                        currA[i] = A[thread + (i * m)];
                    }
                }
            }
            // Loads tile of B
            if (tid < k) {
                #pragma unroll
                for (int i = 0; i < t2; ++i) {
                    if (p + i < n) {
                        currB[tid + (i * t1)] = B[tid + ((p + i) * k)];
                    }
                }
            }

            // Outer product loop
            for (int j = 0; j < k; j += t1) {
                __syncthreads();
                // Loads next tile of B
                if (j + t1 + tid < k) {
                    #pragma unroll
                    for (int i = 0; i < t2; ++i) {
                        if (p + i < n) {
                            nextB[i] = B[(j + t1 + tid) + ((p + i) * k)];
                        }
                    }
                }

                const int t3mod = t1 % t3;

                // Loop over A's columns
                for (int l = j; l < j + (t1 - t3mod) && l < k; l += t3) {
                    // Loads next A
                    #pragma unroll
                    for (int i = 0; i < t3; ++i) {
                        if (l + t3 + i < k && thread < m) {
                            nextA[i] = A[thread + ((l + t3 + i) * m)];
                        }
                    }

                    // Floating Point Operations (lines 32-34)
                    // Each thread does t2 * t3 mults

                    // Either dispatch guarded or unguarded instructions based
                    // on position in matrix A
                    if (l + t3 <= k) {
                        // It is assumed that B[(l - j) .. (l - j) + t3 - 1, _]
                        // exist
                        #pragma unroll
                        for (int a = 0; a < t2; ++a) {
                            #pragma unroll
                            for (int b = 0; b < t3; ++b) {
                                currC[a] +=
                                    currA[b] * currB[(l - j) + b + (a * t1)];
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int a = 0; a < t2; ++a) {
                            #pragma unroll
                            for (int b = 0; b < t3; ++b) {
                                if (l + b < k) {
                                    currC[a] += currA[b] *
                                                currB[(l - j) + b + (a * t1)];
                                }
                            }
                        }
                    }

                    // Stores next A in curr A
                    #pragma unroll
                    for (int i = 0; i < t3; ++i) {
                        currA[i] = nextA[i];
                    }
                }
                // Accommodates t3 that do not divide t1.
                #pragma unroll
                for (int a = 0; a < t2; ++a) {
                    #pragma unroll
                    for (int b = 0; b < t3mod; ++b) {
                        if (j + t1 - t3mod + b < k) {
                            currC[a] +=
                                currA[b] * currB[(t1 - t3mod + b) + (a * t1)];
                        }
                    }
                }

                __syncthreads();

                // Loads currB from each thread's nextB
                #pragma unroll
                for (int i = 0; i < t2; ++i) {
                    currB[tid + (i * t1)] = nextB[i];
                }

                // Loads next currA
                if (t3mod != 0) {
                    #pragma unroll
                    for (int i = 0; i < t3; ++i) {
                        if (j + t1 + i < k && thread < m) {
                            currA[i] = A[thread + ((j + t1 + i) * m)];
                        }
                    }
                }
            }
            // Stores C
            if (thread < m) {
                #pragma unroll
                for (int i = 0; i < t2; ++i) {
                    if (p + i < n) {
                        C[thread + ((p + i) * m)] = currC[i];
                    }
                }
            }
        }
    }
}
 
#endif