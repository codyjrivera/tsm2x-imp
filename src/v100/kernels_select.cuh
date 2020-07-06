/**
 * Kernel selection for V100. This implements the functions declared in kernels.cuh.
 * by Cody Rivera, 2019-2020
 */

#ifndef _KERNELS_SELECT_CUH
#define _KERNELS_SELECT_CUH

#define FLOAT_T1 128
#define DOUBLE_T1 128

/**
 * TSM2 Parameter Choice for V100:
 *
 * t1 := 128
 *
 * Single Precision: n ~= t2, t3 := 32
 * Double Precision: n ~= t2, t3 := 16 if m < 10240, and t3 := 12 otherwise
 */

template <>
void launchKernelTsm2(const float* devA, const float* devB, float* devC,
                      const unsigned int m, const unsigned int n, 
                      const unsigned int k) {
    int blocks = (m / FLOAT_T1) + 1;
    blocks = (blocks > 65536) ? 65536 : blocks;

    if (n <= 2) {
        kernelTsm2<float, FLOAT_T1, 2, 32>
            <<<blocks, FLOAT_T1>>>(devA, devB, devC, m, n, k);
    } else if (n <= 4) {
        kernelTsm2<float, FLOAT_T1, 4, 32>
            <<<blocks, FLOAT_T1>>>(devA, devB, devC, m, n, k);
    } else if (n <= 6) {
        kernelTsm2<float, FLOAT_T1, 6, 32>
            <<<blocks, FLOAT_T1>>>(devA, devB, devC, m, n, k);
    } else if (n <= 8) {
        kernelTsm2<float, FLOAT_T1, 8, 32>
            <<<blocks, FLOAT_T1>>>(devA, devB, devC, m, n, k);
    } else {
        kernelTsm2<float, FLOAT_T1, 16, 32>
            <<<blocks, FLOAT_T1>>>(devA, devB, devC, m, n, k);
    }
}

template <>
void launchKernelTsm2(const double* devA, const double* devB, double* devC,
                      const unsigned int m, const unsigned int n, 
                      const unsigned int k) {
    int blocks = (m / DOUBLE_T1) + 1;
    blocks = (blocks > 65536) ? 65536 : blocks;
    
    if (n <= 2) {
        if (m < 20480) {
            kernelTsm2<double, DOUBLE_T1, 2, 16>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        } else {
            kernelTsm2<double, DOUBLE_T1, 2, 12>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        }
    } else if (n <= 4) {
        if (m < 20480) {
            kernelTsm2<double, DOUBLE_T1, 4, 16>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        } else {
            kernelTsm2<double, DOUBLE_T1, 4, 12>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        }
    } else if (n <= 6) {
        if (m < 20480) {
            kernelTsm2<double, DOUBLE_T1, 6, 16>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        } else {
            kernelTsm2<double, DOUBLE_T1, 6, 12>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        }
    } else if (n <= 8) {
        if (m < 20480) {
            kernelTsm2<double, DOUBLE_T1, 8, 16>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        } else {
            kernelTsm2<double, DOUBLE_T1, 8, 12>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        }
    } else if (n <= 16) {
        if (m < 20480) {
            kernelTsm2<double, DOUBLE_T1, 16, 16>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        } else {
            kernelTsm2<double, DOUBLE_T1, 16, 12>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        }
    } else {
        kernelTsm2<double, DOUBLE_T1, 32, 12>
            <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
    }
}


/**
 * ISM2 Parameter Choice for V100:
 * This is a balancing act between the latency of having many thread
 * blocks and the latency of iterating over different partitions of the
 * matrices with fewer threads.
 * 
 * We introduce a variable tcf, which is the factor that the thread count
 * will be divided by. Experimental evaluation yields the following optimal
 * tcf values for m being 10^4, 10^5, 10^6, and 10^7:
 *
 * Single: 1, 1, 2, 8
 * Double: 1, 1, 1, 4
 * 
 * Furthermore, single precision is better suited to ISM2 optimization strategy 1,
 * which reuses TSM2's kernel with fewer threads, while double precision is better
 * suited to ISM2 optimization strategy 2, with a separate kernel.
 */

template <>
void launchKernelIsm2(const float* devA, const float* devB, float* devC,
                      const unsigned int m, const unsigned int n, 
                      const unsigned int k) {
    int blocks = (m / FLOAT_T1) + 1;
    int tcf = 1;

    // Determine tcf
    if (m >= 1e6 && m < 1e7) {
        tcf = 2;
    } else if (m >= 1e7) {
        tcf = 8;
    }

    blocks /= tcf;
    // Clamp between 1 and 65536
    blocks = (blocks < 1) ? 1 : blocks;
    blocks = (blocks > 65536) ? 65536 : blocks;

    if (n <= 2) {
        kernelTsm2<float, FLOAT_T1, 2, 32>
            <<<blocks, FLOAT_T1>>>(devA, devB, devC, m, n, k);
    } else if (n <= 4) {
        kernelTsm2<float, FLOAT_T1, 4, 32>
            <<<blocks, FLOAT_T1>>>(devA, devB, devC, m, n, k);
    } else if (n <= 6) {
        kernelTsm2<float, FLOAT_T1, 6, 32>
            <<<blocks, FLOAT_T1>>>(devA, devB, devC, m, n, k);
    } else if (n <= 8) {
        kernelTsm2<float, FLOAT_T1, 8, 32>
            <<<blocks, FLOAT_T1>>>(devA, devB, devC, m, n, k);
    } else {
        kernelTsm2<float, FLOAT_T1, 16, 32>
            <<<blocks, FLOAT_T1>>>(devA, devB, devC, m, n, k);
    }
}
                      
template <>
void launchKernelIsm2(const double* devA, const double* devB, double* devC,
                      const unsigned int m, const unsigned int n, 
                      const unsigned int k) {
    int blocks = (m / DOUBLE_T1) + 1;
    int tcf = 1;

    // Determine tcf
    if (m >= 1e7) {
        tcf = 4;
    }

    blocks /= tcf;
    // Clamp between 1 and 65536
    blocks = (blocks < 1) ? 1 : blocks;
    blocks = (blocks > 65536) ? 65536 : blocks;

    // Workaround for Opt2 Bug
    if (k > DOUBLE_T1) {
        if (n <= 2) {
            if (m < 20480) {
                kernelTsm2<double, DOUBLE_T1, 2, 16>
                    <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
            } else {
                kernelTsm2<double, DOUBLE_T1, 2, 12>
                    <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
            }
        } else if (n <= 4) {
            if (m < 20480) {
                kernelTsm2<double, DOUBLE_T1, 4, 16>
                    <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
            } else {
                kernelTsm2<double, DOUBLE_T1, 4, 12>
                    <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
            }
        } else if (n <= 6) {
            if (m < 20480) {
                kernelTsm2<double, DOUBLE_T1, 6, 16>
                    <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
            } else {
                kernelTsm2<double, DOUBLE_T1, 6, 12>
                    <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
            }
        } else if (n <= 8) {
            if (m < 20480) {
                kernelTsm2<double, DOUBLE_T1, 8, 16>
                    <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
            } else {
                kernelTsm2<double, DOUBLE_T1, 8, 12>
                    <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
            }
        } else if (n <= 16) {
            if (m < 20480) {
                kernelTsm2<double, DOUBLE_T1, 16, 16>
                    <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
            } else {
                kernelTsm2<double, DOUBLE_T1, 16, 12>
                    <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
            }
        } else {
            kernelTsm2<double, DOUBLE_T1, 32, 12>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        }
        return;
    }
    if (n <= 2) {
        if (m < 20480) {
            kernelIsm2Opt2<double, DOUBLE_T1, 2, 16>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        } else {
            kernelIsm2Opt2<double, DOUBLE_T1, 2, 12>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        }
    } else if (n <= 4) {
        if (m < 20480) {
            kernelIsm2Opt2<double, DOUBLE_T1, 4, 16>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        } else {
            kernelIsm2Opt2<double, DOUBLE_T1, 4, 12>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        }
    } else if (n <= 6) {
        if (m < 20480) {
            kernelIsm2Opt2<double, DOUBLE_T1, 6, 16>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        } else {
            kernelIsm2Opt2<double, DOUBLE_T1, 6, 12>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        }
    } else if (n <= 8) {
        if (m < 20480) {
            kernelIsm2Opt2<double, DOUBLE_T1, 8, 16>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        } else {
            kernelIsm2Opt2<double, DOUBLE_T1, 8, 12>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        }
    } else if (n <= 16) {
        if (m < 20480) {
            kernelIsm2Opt2<double, DOUBLE_T1, 16, 16>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        } else {
            kernelIsm2Opt2<double, DOUBLE_T1, 16, 12>
                <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
        }
    } else {
        kernelIsm2Opt2<double, DOUBLE_T1, 32, 12>
            <<<blocks, DOUBLE_T1>>>(devA, devB, devC, m, n, k);
    }
}
                                 


#endif