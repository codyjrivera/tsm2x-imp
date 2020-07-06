/**
 * CUBLAS wrapper for different matrix types
 * by Cody Rivera, 2019-2020
 */

#include "cublas_v2.h"
#include "launch_cublas.cuh"


// float specialization
template <>
cublasStatus_t launchCublas(cublasHandle_t handle, float& one, float& zero,
                            const float* devA, const float* devB, float* devC,
                            const unsigned int m, const unsigned int n, 
                            const unsigned int k) {
    return cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one,
                        devA, CUDA_R_32F, m, devB, CUDA_R_32F, k, &zero,
                        devC, CUDA_R_32F, m, CUDA_R_32F,
                        CUBLAS_GEMM_DEFAULT);
}

// double specialization
template <>
cublasStatus_t launchCublas(cublasHandle_t handle, double& one, double& zero,
                            const double* devA, const double* devB, double* devC,
                            const unsigned int m, const unsigned int n, 
                            const unsigned int k) {
    return cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one,
                        devA, CUDA_R_64F, m, devB, CUDA_R_64F, k, &zero,
                        devC, CUDA_R_64F, m, CUDA_R_64F,
                        CUBLAS_GEMM_DEFAULT);
}
 