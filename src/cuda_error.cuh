/**
 * CUDA and CUBLAS error handling macros
 */

#ifndef _CUDA_ERROR_H
#define _CUDA_ERROR_H

#include <cstdio>

#include "cublas_v2.h"
#include "cuda_runtime.h"

// Useful macros
#define cudaErrchk(ans) \
    { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char* file, int line,
                       bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDAassert: %s %s %d\n", cudaGetErrorString(code),
                file, line);
        if (abort) exit(code);
    }
}

#define cublasErrchk(ans) \
    { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char* file, int line,
                         bool abort = true) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLASassert Failure Code: %d %s %d\n", code, file,
                line);
        if (abort) exit(code);
    }
}

#endif