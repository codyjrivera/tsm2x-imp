/**
 * Matrix Generation Program
 * by Cody Rivera, 2019
 * 
 * Usage - ./gen [-d] -r ROW_COUNT -c COL_COUNT file
 */

#define FLOAT_MAX 500

#include <fstream>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <ctime>

template <typename FloatType>
void buildMatrix(FloatType *A, unsigned int m, unsigned int n) {
    srand(time(NULL));
    unsigned int iMax = m * n;
    for (unsigned int i = 0; i < iMax; i++) {
        FloatType raw = (FloatType)rand() / (FloatType)RAND_MAX;
        A[i] = FLOAT_MAX * raw;
    }
}

int main(int argc, char **argv) {
    unsigned int m = 0, n = 0;
    int fileArg = 0;
    int FloatTypeSize = sizeof(float);
    for (int i = 1; i < argc; ++i) {
        if (strcmp("-d", argv[i]) == 0) {
            FloatTypeSize = sizeof(double);
        } else if (strcmp("-r", argv[i]) == 0) {
            if (i + 1 < argc) {
                m = atol(argv[i + 1]);
                ++i;
            }
        } else if (strcmp("-c", argv[i]) == 0) {
            if (i + 1 < argc) {
                n = atol(argv[i + 1]);
                ++i;
            }
        } else {
            fileArg = i;
        }
    }

    if (m == 0 || n == 0 || fileArg == 0) {
        std::cerr << "Usage: " << argv[0] << " [-d] -r ROW_COUNT -c COL_COUNT file"
                            << std::endl;
        return 1;
    }
    std::ofstream binFile(argv[fileArg], std::ios::binary);
    if (!binFile) {
        std::cerr << "Cannot open " << argv[fileArg] << " for reading" << std::endl;
        return 1;
    }

    binFile.write((char *)&m, sizeof(unsigned int));
    binFile.write((char *)&n, sizeof(unsigned int));

    // Raw array
    char *A = new char[(size_t)m * (size_t)n * (size_t)FloatTypeSize];

    if (FloatTypeSize == sizeof(float)) {
        buildMatrix<float>((float *)A, m, n);
    } else {
        buildMatrix<double>((double *)A, m, n);
    }

    binFile.write((char *)A, (size_t)m * (size_t)n * (size_t)FloatTypeSize);
    binFile.close();

    delete[] A;
}
