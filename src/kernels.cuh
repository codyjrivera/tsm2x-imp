/**
 * External interfaces to TSM2/ISM2 kernels
 * by Cody Rivera, 2019-2020
 */

#ifndef _KERNELS_CUH
#define _KERNELS_CUH

template <typename FloatType>
void launchKernelTsm2(const FloatType* devA, const FloatType* devB, FloatType* devC,
                      const unsigned int m, const unsigned int n, 
                      const unsigned int k);

template <typename FloatType>
void launchKernelIsm2(const FloatType* devA, const FloatType* devB, FloatType* devC,
                      const unsigned int m, const unsigned int n, 
                      const unsigned int k);

#endif