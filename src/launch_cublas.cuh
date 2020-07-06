/**
 * External interfaces to CUBLAS
 * by Cody Rivera, 2019-2020
 */

#ifndef _LAUNCH_CUBLAS_CUH
#define _LAUNCH_CUBLAS_CUH
 
#include "cublas_v2.h"

template <typename FloatType>
cublasStatus_t launchCublas(cublasHandle_t handle, FloatType& one, FloatType& zero,
                            const FloatType* devA, const FloatType* devB, FloatType* devC,
                            const unsigned int m, const unsigned int n, 
                            const unsigned int k);
 
#endif