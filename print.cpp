/*
  Matrix Print Program - by Cody Rivera
  Usage - ./a.out file
 */

#define COL_WIDTH 8

#include <fstream>
#include <iostream>
#include <iomanip>


int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] 
                  << " matrixFile (litte endian)" << std::endl;
        return 1;
    }
    std::ifstream binFile(argv[1], std::ios::binary);
    if (!binFile)
    {
        std::cerr << "Cannot open " << argv[1]
                  << " for reading" << std::endl;
        return 1;
    }
    unsigned int m, n;
    binFile.read((char*)&m, sizeof(unsigned int));
    binFile.read((char*)&n, sizeof(unsigned int));
    
    float* A = new float[m * n];
    binFile.read((char*)A, m * n * sizeof(float));
    binFile.close();
    
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
    
    delete[] A;
}

