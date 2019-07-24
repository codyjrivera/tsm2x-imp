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


bool matrixCompare(const float* A, const float* B,
                   unsigned int m, unsigned int n,
                   unsigned int& iFail, unsigned int& jFail)
{
    float aVal, bVal;
    // Cache-friendly comparison pattern
    for (unsigned int j = 0; j < n; j++)
    {
        for (unsigned int i = 0; i < m; i++)
        {
            aVal = A[i + (j * m)];
            bVal = B[i + (j * m)];
            if (fabs(aVal - bVal) > EPS)
            {
                iFail = i;
                jFail = j;
                return false;
            }
        }
    }
    return true;
}

void reportTestSuccess(const char* testName, double GFLOPs)
{
    printf("%s succeeded: %g GFLOPs\n", testName, GFLOPs);
}

void reportTestFailure(const char* testName,
                       const float* orig, const float* cand,
                       unsigned int leadDim,
                       unsigned int iFail, unsigned int jFail)
{
    double oVal = orig[iFail + (jFail * leadDim)];
    double cVal = cand[iFail + (jFail * leadDim)];
    fprintf(stderr, "%s failed: Original[%u, %u] = %.6f != Candidate[%u, %u] = %.6f\n",
            testName, iFail, jFail, oVal, iFail, jFail, cVal);
}

double getGFLOPs(double time, unsigned int m, unsigned int n, unsigned int k)
{
    double instCount = ((double) m * (double) n * (double) k) / 10e9;
    double timeSeconds = time / 1000;
    return instCount / timeSeconds;
}




/*
  Executes the kernels
 */
bool runKernels(const float* A, const float* B, float* C,
                const unsigned int m, const unsigned int n,
                const unsigned int k)
{
    // Candidate for C -- Used by GPU kernels
    float* candC;
    // Device memory
    float* devA, * devB, * devC;
    // Events used for timing
    cudaEvent_t start, end;
    float time;

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
    
    // Cuda Memory Copy
    cudaErrchk(cudaMemcpy(devA, A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(devB, B, k * n * sizeof(float), cudaMemcpyHostToDevice));

    // Inits CUDA events
    cudaErrchk(cudaEventCreate(&start));
    cudaErrchk(cudaEventCreate(&end));
    
    // Runs CUBLAS call
    cublasHandle_t handle;
    cublasErrchk(cublasCreate(&handle));
    
    float one = 1;
    float zero = 0;

    cudaErrchk(cudaEventRecord(start));
    cublasErrchk(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k,
                             &one, devA, m, devB, k,
                             &zero, devC, m));
    cudaErrchk(cudaEventRecord(end));
    
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaEventElapsedTime(&time, start, end));
    
    reportTestSuccess("CUBLAS Test", getGFLOPs(time, m, n, k)); 
    // Copies result back
    cudaErrchk(cudaMemcpy(C, devC, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    cublasErrchk(cublasDestroy(handle));
    
    // Runs kernels
    // Failure flag
    bool status;
    // Failure indices
    unsigned int iFail, jFail;
    // Calculates tile numbers
    unsigned int blocksX = (m / TILE_WIDTH) + 1;
    unsigned int blocksY = (n / TILE_WIDTH) + 1;
    if (blocksX > MAX_TILES)
    {
        blocksX = MAX_TILES;
    }
    if (blocksY > MAX_TILES)
    {
        blocksY = MAX_TILES;
    }

    dim3 numBlocks(blocksX, blocksY), blockSize(TILE_WIDTH, TILE_WIDTH);

    // Naive Kernel
    cudaErrchk(cudaEventRecord(start));
    naiveGEMMKernel<<<numBlocks, blockSize>>>(devA, devB, devC, m, n, k);
    cudaErrchk(cudaEventRecord(end));
    cudaErrchk(cudaGetLastError());

    // Timing
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaEventElapsedTime(&time, start, end));
    
    // Copying, checking and reporting
    cudaErrchk(cudaMemcpy(candC, devC, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    status = matrixCompare(C, candC, m, n, iFail, jFail);
    if (status)
    {
        reportTestSuccess("Naive test", getGFLOPs(time, m, n, k));
    }
    else
    {
        reportTestFailure("Naive test", C, candC, m, iFail, jFail);
        return false;
    }


    // Shared Kernel
    size_t sharedSize = 2 * ((blockSize.x * blockSize.y) + blockSize.y);
    cudaErrchk(cudaEventRecord(start));
    sharedGEMMKernel<<<numBlocks, blockSize, sharedSize * sizeof(float)>>>(devA, devB, devC, m, n, k);
    cudaErrchk(cudaEventRecord(end));
    cudaErrchk(cudaGetLastError());

    // Timing
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaEventElapsedTime(&time, start, end));
    
    // Copying, checking and reporting
    cudaErrchk(cudaMemcpy(candC, devC, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    status = matrixCompare(C, candC, m, n, iFail, jFail);
    if (status)
    {
        reportTestSuccess("Shared test", getGFLOPs(time, m, n, k));
    }
    else
    {
        reportTestFailure("Shared test", C, candC, m, iFail, jFail);
        return false;
    }

    // Opt Kernel
    cudaErrchk(cudaEventRecord(start));
    optGEMMKernel<<<numBlocks, blockSize, sharedSize * sizeof(float)>>>(devA, devB, devC, m, n, k);
    cudaErrchk(cudaEventRecord(end));
    cudaErrchk(cudaGetLastError());

    // Timing
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaEventElapsedTime(&time, start, end));
    
    // Copying, checking and reporting
    cudaErrchk(cudaMemcpy(candC, devC, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    status = matrixCompare(C, candC, m, n, iFail, jFail);
    if (status)
    {
        reportTestSuccess("Optimal test", getGFLOPs(time, m, n, k));
    }
    else
    {
        reportTestFailure("Optimal test", C, candC, m, iFail, jFail);
        return false;
    }

    // Deletes memory
    cudaErrchk(cudaEventDestroy(start));
    cudaErrchk(cudaEventDestroy(end));
    free(candC);
    cudaErrchk(cudaFree(devA));
    cudaErrchk(cudaFree(devB));
    cudaErrchk(cudaFree(devC));
    
    return true;
}