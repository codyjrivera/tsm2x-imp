/**
 * Testbed declarations
 * by Cody Rivera, 2019-2020
 */

#ifndef _MULTIPLY_CUH
#define _MULTIPLY_CUH

// Functions
template<typename FloatType>
bool runKernels(const FloatType* A, const FloatType* B, FloatType* C,
                const unsigned int m, const unsigned int n,
                const unsigned int k, const bool runIsm2);

#endif