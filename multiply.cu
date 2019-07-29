/*
  multiply.cu -- Matrix multiplication testbench - by Cody Rivera
*/

#include <cstdio>
#include <cstdlib>
#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "multiply.cuh"

#define EPS 10e-2
#define MAX_TILES 255

/*
  Helper functions
 */


template<typename FloatType>
bool matrixCompare(const FloatType* A, const FloatType* B,
                   unsigned int m, unsigned int n,
                   unsigned int& iFail, unsigned int& jFail)
{
    FloatType aVal, bVal;
    bool b = true;
    // Cache-friendly comparison pattern
    for (unsigned int j = 0; j < n && b; j++)
    {
        for (unsigned int i = 0; i < m && b; i++)
        {
            aVal = A[i + (j * m)];
            bVal = B[i + (j * m)];
            if (fabs(aVal - bVal) > EPS)
            {
                iFail = i;
                jFail = j;
                b = false;
            }
        }
    }
    return b;
}

template<typename FloatType>
void reportTestSuccess(const char* testName, double GFLOPs, double totalGFLOPs)
{
    printf("%s succeeded: %g GFLOPs, %g GFLOPs acc. for transfers\n", testName, GFLOPs, totalGFLOPs);
}

template<typename FloatType>
void reportTestFailure(const char* testName,
                       const FloatType* orig, const FloatType* cand,
                       unsigned int leadDim,
                       unsigned int iFail, unsigned int jFail)
{
    double oVal = (double)orig[iFail + (jFail * leadDim)];
    double cVal = (double)cand[iFail + (jFail * leadDim)];
    fprintf(stderr, "%s failed: Original[%u, %u] = %.6f != Candidate[%u, %u] = %.6f\n",
            testName, iFail, jFail, oVal, iFail, jFail, cVal);
}

template<typename FloatType>
double getGFLOPs(double time, unsigned int m, unsigned int n, unsigned int k)
{
    double instCount = ((double) m * (double) n * (double) k) / 10e9;
    double timeSeconds = time / 1000;
    return instCount / timeSeconds;
}




/*
  Executes the kernels
 */
template<>
bool runKernels(const float* A, const float* B, float* C,
                const unsigned int m, const unsigned int n,
                const unsigned int k)
{
    // Candidate for C -- Used by GPU kernels
    float* candC;
    // Device memory
    float* devA, * devB, * devC;
    // Events used for timing
    cudaEvent_t start, end, startTotal, endTotal;
    float time, timeTotal;

    printf("Multiplying matrix A[%u, %u] by matrix B[%u, %u]\n\n", m, k, k, n); 

    // Allocates new memory
    candC = (float*)malloc(m * n * sizeof(float));
    if (candC == NULL)
    {
        fprintf(stderr, "Not enough memory\n");
        return false;
    }
    
    cudaErrchk(cudaMalloc((float**)&devA, m * k * sizeof(float)));
    cudaErrchk(cudaMalloc((float**)&devB, k * n * sizeof(float)));
    cudaErrchk(cudaMalloc((float**)&devC, m * n * sizeof(float)));
    
    
    // Inits CUDA events
    cudaErrchk(cudaEventCreate(&start));
    cudaErrchk(cudaEventCreate(&end));
    cudaErrchk(cudaEventCreate(&startTotal));
    cudaErrchk(cudaEventCreate(&endTotal));
    
    // Runs CUBLAS call
    cublasHandle_t handle;
    cublasErrchk(cublasCreate(&handle));
    
    float one = 1;
    float zero = 0;

    cudaErrchk(cudaEventRecord(startTotal));
    
    // Cuda Memory Copy
    cudaErrchk(cudaMemcpy(devA, A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(devB, B, k * n * sizeof(float), cudaMemcpyHostToDevice));

    cudaErrchk(cudaEventRecord(start));
    cublasErrchk(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k,
                             &one, devA, m, devB, k,
                             &zero, devC, m));
    cudaErrchk(cudaEventRecord(end));
    
    // Copies result back
    cudaErrchk(cudaMemcpy(C, devC, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    cudaErrchk(cudaEventRecord(endTotal));
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaEventElapsedTime(&time, start, end));
    cudaErrchk(cudaEventElapsedTime(&timeTotal, startTotal, endTotal));
    reportTestSuccess<float>("CUBLAS Test", getGFLOPs<float>(time, m, n, k), getGFLOPs<float>(timeTotal, m, n, k)); 

    cublasErrchk(cublasDestroy(handle));
    

    /*
    // Runs kernels
    // Failure flag
    bool status;
    // Failure indices
    unsigned int iFail, jFail;
    */

    cudaErrchk(cudaEventDestroy(start));
    cudaErrchk(cudaEventDestroy(end));
    cudaErrchk(cudaEventDestroy(startTotal));
    cudaErrchk(cudaEventDestroy(endTotal));
    free(candC);
    cudaErrchk(cudaFree(devA));
    cudaErrchk(cudaFree(devB));
    cudaErrchk(cudaFree(devC));
    
    return true;
}




template<>
bool runKernels(const double* A, const double* B, double* C,
                const unsigned int m, const unsigned int n,
                const unsigned int k)
{
    // Candidate for C -- Used by GPU kernels
    double* candC;
    // Device memory
    double* devA, * devB, * devC;
    // Events used for timing
    cudaEvent_t start, end, startTotal, endTotal;
    float time, timeTotal;

    printf("Multiplying matrix A[%u, %u] by matrix B[%u, %u]\n\n", m, k, k, n); 

    // Allocates new memory
    candC = (double*)malloc(m * n * sizeof(double));
    if (candC == NULL)
    {
        fprintf(stderr, "Not enough memory\n");
        return false;
    }
    
    cudaErrchk(cudaMalloc((double**)&devA, m * k * sizeof(double)));
    cudaErrchk(cudaMalloc((double**)&devB, k * n * sizeof(double)));
    cudaErrchk(cudaMalloc((double**)&devC, m * n * sizeof(double)));
    
    
    // Inits CUDA events
    cudaErrchk(cudaEventCreate(&start));
    cudaErrchk(cudaEventCreate(&end));
    cudaErrchk(cudaEventCreate(&startTotal));
    cudaErrchk(cudaEventCreate(&endTotal));
    
    // Runs CUBLAS call
    cublasHandle_t handle;
    cublasErrchk(cublasCreate(&handle));
    
    double one = 1;
    double zero = 0;

    cudaErrchk(cudaEventRecord(startTotal));
    
    // Cuda Memory Copy
    cudaErrchk(cudaMemcpy(devA, A, m * k * sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(devB, B, k * n * sizeof(double), cudaMemcpyHostToDevice));

    cudaErrchk(cudaEventRecord(start));
    cublasErrchk(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k,
                             &one, devA, m, devB, k,
                             &zero, devC, m));
    cudaErrchk(cudaEventRecord(end));
    
    // Copies result back
    cudaErrchk(cudaMemcpy(C, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost));

    cudaErrchk(cudaEventRecord(endTotal));
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaEventElapsedTime(&time, start, end));
    cudaErrchk(cudaEventElapsedTime(&timeTotal, startTotal, endTotal));
    reportTestSuccess<double>("CUBLAS Test", getGFLOPs<double>(time, m, n, k), getGFLOPs<double>(timeTotal, m, n, k)); 

    cublasErrchk(cublasDestroy(handle));
    

    /*
    // Runs kernels
    // Failure flag
    bool status;
    // Failure indices
    unsigned int iFail, jFail;
    */

    cudaErrchk(cudaEventDestroy(start));
    cudaErrchk(cudaEventDestroy(end));
    cudaErrchk(cudaEventDestroy(startTotal));
    cudaErrchk(cudaEventDestroy(endTotal));
    free(candC);
    cudaErrchk(cudaFree(devA));
    cudaErrchk(cudaFree(devB));
    cudaErrchk(cudaFree(devC));
    
    return true;
}