/*
  main.cpp -- Handles IO of matrices, calls the actual program
  Written by Cody Rivera
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>

template<typename FloatType>
bool runKernels(const FloatType* A, const FloatType* B, FloatType* C,
                const unsigned int m, const unsigned int n,
                const unsigned int k);

template<typename FloatType>
bool runMatmul(std::istream& fileA, std::istream& fileB, std::ostream& outFile)
{
    FloatType* A, * B, * C;
    int m, n, k, kCand;

    // Reads Matrix Sizes
    fileA.read((char*)&m, sizeof(unsigned int));
    fileA.read((char*)&k, sizeof(unsigned int));
    fileB.read((char*)&kCand, sizeof(unsigned int));
    fileB.read((char*)&n, sizeof(unsigned int));

    if (k != kCand)
    {
        fprintf(stderr, 
                "Matrix multiplication is undefined where A's"
                "column count is not equal\n to B's row count\n\n"
                "Matrix A (%u x %u) and Matrix B (%u x %u)\n",
                m, k, kCand, n);
        return false;
    }
    
    // Mallocs Matrices on CPU
    A = (FloatType*) malloc(m * k * sizeof(FloatType));
    B = (FloatType*) malloc(k * n * sizeof(FloatType));
    C = (FloatType*) malloc(m * n * sizeof(FloatType));

    if (A == NULL || B == NULL || C == NULL)
    {
        fprintf(stderr, "Not enough memory\n");
        return false;
    }

    // Loads Data to Matrix A and B
    fileA.read((char*)A, m * k * sizeof(FloatType));
    fileB.read((char*)B, k * n * sizeof(FloatType));
    /*
    for (unsigned int j = 0; j < k; j++)
    {
        for (unsigned int i = 0; i < m; i++)
        {
            fileA.read((char*)&A[i + (j * m)], sizeof(FloatType)); 
        }
    }

    for (unsigned int j = 0; j < n; j++)
    {
        for (unsigned int i = 0; i < k; i++)
        {
            fileB.read((char*)&B[i + (j * k)], sizeof(FloatType)); 
        }
    }
    */

    // Calls CUDA
    bool status = runKernels<FloatType>(A, B, C, m, n, k);
    if (!status)
    {
        free(A); 
        free(B); 
        free(C);
        return false;
    }
    
    // Writes output matrix
    outFile.write((const char*)&m, sizeof(unsigned int));
    outFile.write((const char*)&n, sizeof(unsigned int));
    
    outFile.write((const char*)C, m * n * sizeof(FloatType));
    return true;
}

int main(int argc, char** argv)
{
    int fileArg[3];
    int nFiles = 0;
    bool isDouble = false;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-d") == 0)
        {
            isDouble = true;
        }
        else
        {
            if (nFiles < 3)
            {
                fileArg[nFiles] = i;
            }
            nFiles++;
        }
    }
    if (nFiles != 3)
    {
        fprintf(stderr, "Usage: %s [-d] matrixA matrixB matrixC\n", argv[0]);
        return 1;
    }
    // Using C++ fileio because I don't want to have to take care of
    // closing files
    std::ifstream fileA(argv[fileArg[0]], std::ios::binary), fileB(argv[fileArg[1]], std::ios::binary);
    std::ofstream outFile(argv[fileArg[2]], std::ios::binary);
    if (!fileA)
    {
        fprintf(stderr, "Cannot open %s for reading\n", argv[1]);
        return 1;
    }
    if (!fileB)
    {
        fprintf(stderr, "Cannot open %s for reading\n", argv[2]);
        return 1;
    }
    if (!outFile)
    {
        fprintf(stderr, "Cannot open %s for writing\n", argv[3]);
        return 1;
    }
    // Runs matmul
    bool status = false;
    if (isDouble)
    {
        status = runMatmul<double>(fileA, fileB, outFile);
    }
    else
    {
        status = runMatmul<float>(fileA, fileB, outFile);
    }
    fileA.close();
    fileB.close();
    outFile.close();
    if (status)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}
