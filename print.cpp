/*
  Matrix Print Program - by Cody Rivera
  Usage - ./a.out file
 */

#define COL_WIDTH 8

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>


template<typename FloatType>
void printMatrix(FloatType* A, unsigned int m, unsigned int n)
{
    std::cout << std::setw(COL_WIDTH) << m << " "
              << std::setw(COL_WIDTH) << n << std::endl << std::endl;
    
    for (unsigned int i = 0; i < m; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            std::cout << std::setw(COL_WIDTH) << A[i + (j * m)] << " ";
        }
        std::cout << std::endl;
    }
}


int main(int argc, char** argv)
{
    int fileArg = 0;
    int FloatTypeSize = sizeof(float);
    for (int i = 1; i < argc; i++)
    {
        if (strcmp("-d", argv[i]) != 0)
        {
            FloatTypeSize = sizeof(double);
        }
        else
        {
            fileArg = i;
        }
    }
    if (fileArg == 0)
    {
        std::cerr << "Usage: " << argv[0] 
                  << " [-d] matrixFile (litte endian)" << std::endl;
        return 1;
    }
    std::ifstream binFile(argv[fileArg], std::ios::binary);
    if (!binFile)
    {
        std::cerr << "Cannot open " << argv[fileArg]
                  << " for reading" << std::endl;
        return 1;
    }
    unsigned int m, n;
    binFile.read((char*)&m, sizeof(unsigned int));
    binFile.read((char*)&n, sizeof(unsigned int));
    
    // Raw array
    char* A = new char[m * n * FloatTypeSize];
    binFile.read((char*)A, m * n * FloatTypeSize);
    binFile.close();

    if (FloatTypeSize == sizeof(float))
    {
        printMatrix<float>((float*) A, m, n);
    }
    else
    {
        printMatrix<double>((double*) A, m, n);
    }

    delete[] A;
}

