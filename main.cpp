/*
  main.cpp -- Handles IO of matrices, calls the actual program
  Written by Cody Rivera
*/

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>

// Supplied in CUDA file
extern bool runKernels(const float* A, const float* B, float* C,
                       const unsigned int m, const unsigned int n,
                       const unsigned int k);


int main(int argc, char** argv)
{
    float* A, * B, * C;
    unsigned int m, n, k, kCand;
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s matrixA matrixB matrixC\n", argv[0]);
        return 1;
    }
    // Using C++ fileio because I don't want to have to take care of
    // closing files
    std::ifstream fileA(argv[1], std::ios::binary), fileB(argv[2], std::ios::binary);
    std::ofstream outFile(argv[3], std::ios::binary);
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
        return 1;
    }
    
    // Mallocs Matrices on CPU
    A = (float*) malloc(m * k * sizeof(float));
    B = (float*) malloc(k * n * sizeof(float));
    C = (float*) malloc(m * n * sizeof(float));

    // Loads Data to Matrix A and B
    fileA.read((char*)A, m * k * sizeof(float));
    fileB.read((char*)B, k * n * sizeof(float));
    /*
    for (unsigned int j = 0; j < k; j++)
    {
        for (unsigned int i = 0; i < m; i++)
        {
            fileA.read((char*)&A[i + (j * m)], sizeof(float)); 
        }
    }

    for (unsigned int j = 0; j < n; j++)
    {
        for (unsigned int i = 0; i < k; i++)
        {
            fileB.read((char*)&B[i + (j * k)], sizeof(float)); 
        }
    }
    */

    fileA.close(); 
    fileB.close();

    // Calls CUDA
    bool status = runKernels(A, B, C, m, n, k);
    if (!status)
    {
        free(A); 
        free(B); 
        free(C);
        return 1;
    }
    
    // Writes output matrix
    outFile.write((const char*)&m, sizeof(unsigned int));
    outFile.write((const char*)&n, sizeof(unsigned int));
    
    outFile.write((const char*)C, m * n * sizeof(float));
    /*
    for (unsigned int j = 0; j < n; j++)
    {
        for (unsigned int i = 0; i < m; i++)
        {
            outFile << std::setw(8) << C[i + (j * m)] << " ";
        }
        outFile << std::endl;
    }
    */
    outFile.close();
    
    return 0;
}
