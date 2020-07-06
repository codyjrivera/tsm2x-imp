/**
 * Implementation of ISM2's second optimization.
 * BUG: Will not work for any k > t1
 *
 * by Cody Rivera, 2019-2020
 */

#ifndef _KERNEL_ISM2_OPT2_CUH
#define _KERNEL_ISM2_OPT2_CUH

#include "cuda_runtime.h"

template <typename FloatType, int t1, int t2, int t3>
__global__ void kernelIsm2Opt2(const FloatType* A, const FloatType* B,
                            FloatType* C, const unsigned int m,
                            const unsigned int n, const unsigned int k) {
    // Names mostly follow the paper's
    __shared__ FloatType currB[t1 * t2];

    FloatType currA[t3];
    FloatType nextA[t3];
    FloatType nextB[t2];
    FloatType currC[t2];
    FloatType nextC[t2];

    const int tid = threadIdx.x;
    int threadBaseInit = (blockIdx.x * blockDim.x);
    int threadBase = threadBaseInit;
    int thread = threadBase + tid;

    // Initial load - This'll have to be done anyway, as we're scheduling the
    // loads to happen concurrently with computation Loads first tile of output
    // registers and A
    if (thread < m) {
        #pragma unroll
        for (int i = 0; i < t2; ++i) {
            if (i < n) {
                currC[i] = C[thread + (i * m)];
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
            if (i < n) {
                currB[tid + (i * t1)] = B[tid + (i * k)];
            }
        }
    }

    for (int p = 0; p < n; p += t2) {
        // Outer product loop
        for (int j = 0; j < k; j += t1) {
            threadBase = threadBaseInit;
            thread = threadBase + tid;
            __syncthreads();
            // If there are more rows of Matrix B
            if (k > j + t1) {
                #pragma unroll
                for (int i = 0; i < t2; ++i) {
                    if (j + t1 + tid < k && p + i < n) {
                        nextB[i] = B[(j + t1 + tid) + ((p + i) * k)];
                    }
                }
            }
            // If there are more columns of B to consider
            else if (n > p + t2) {
                #pragma unroll
                for (int i = 0; i < t2; ++i) {
                    if (tid < k && p + t2 + i < n) {
                        nextB[i] = B[tid + ((p + t2 + i) * k)];
                    }
                }

                #pragma unroll
                for (int i = 0; i < t2; ++i) {
                    if (p + i + t2 < n && thread < m) {
                        nextC[i] = C[thread + ((p + t2 + i) * m)];
                    }
                }
            }

            const int t3mod = t1 % t3;

            for (; threadBase < m; threadBase += blockDim.x * gridDim.x) {
                thread = threadBase + tid;
                // Loads next C for next rows
                if (m > threadBase + (blockDim.x * gridDim.x)) {
                    #pragma unroll
                    for (int i = 0; i < t2; ++i) {
                        nextC[i] = C[thread + (blockDim.x * gridDim.x) +
                                    ((p + i) * m)];
                    }
                } else if (k > j + t1) {
                    #pragma unroll
                    for (int i = 0; i < t2; ++i) {
                        nextC[i] = C[threadBaseInit + tid + ((p + i) * m)];
                    }
                } else if (n > p + t2) {
                    #pragma unroll
                    for (int i = 0; i < t2; ++i) {
                        nextC[i] = C[threadBaseInit + tid + ((p + t2 + i) * m)];
                    }
                }

                // Loop over A's columns
                for (int l = j; l < j + (t1 - t3mod) && l < k; l += t3) {
                    bool lastIter = false;
                    // Loads next A
                    if (j + (t1 - t3mod) > l + t3 && k > l + t3) {
                        #pragma unroll
                        for (int i = 0; i < t3; ++i) {
                            if (l + t3 + i < k && thread < m) {
                                nextA[i] = A[thread + ((l + t3 + i) * m)];
                            }
                        }
                    }
                    // Loads the first A for the next group of threads
                    else if (m > threadBase + (blockDim.x * gridDim.x)) {
                        #pragma unroll
                        for (int i = 0; i < t3; ++i) {
                            if (i < k &&
                                thread + (blockDim.x * gridDim.x) < m) {
                                nextA[i] = A[thread + (blockDim.x * gridDim.x) +
                                            (i * m)];
                            }
                        }
                    }
                    // Loads start of next tile of A that corresponds with next
                    // tile of B
                    else if (k > j + t1) {
                        lastIter = true;
                        #pragma unroll
                        for (int i = 0; i < t3; ++i) {
                            if (j + t1 + i < k && threadBaseInit + tid < m) {
                                nextA[i] = A[threadBaseInit + tid +
                                            ((j + t1 + i) * m)];
                            }
                        }
                    }
                    // If there are more columns of B to consider
                    else if (n > p + t2) {
                        #pragma unroll
                        for (int i = 0; i < t3; ++i) {
                            if (i < k && threadBaseInit + tid < m) {
                                nextA[i] = A[threadBaseInit + tid + (i * m)];
                            }
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

                    if (lastIter) {
                        // Accommodates t3 that do not divide t1.
                        #pragma unroll
                        for (int a = 0; a < t2; ++a) {
                            #pragma unroll
                            for (int b = 0; b < t3mod; ++b) {
                                if (j + t1 - t3mod + b < k) {
                                    currC[a] +=
                                        currA[b] *
                                        currB[(t1 - t3mod + b) + (a * t1)];
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

                __syncthreads();

                // Stores C
                if (thread < m) {
                    #pragma unroll
                    for (int i = 0; i < t2; ++i) {
                        if (p + i < n) {
                            C[thread + ((p + i) * m)] = currC[i];
                        }
                    }
                }
                // If applicable, loads nextC to currC
                #pragma unroll
                for (int i = 0; i < t2; ++i) {
                    currC[i] = nextC[i];
                }
            }
            // Loads currB from each thread's nextB
            #pragma unroll
            for (int i = 0; i < t2; ++i) {
                currB[tid + (i * t1)] = nextB[i];
            }
        }
    }
}

#endif
 