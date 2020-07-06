/**
 * Kernel implementation and launching.
 * by Cody Rivera, 2019-2020
 */

#include "kernels.cuh"

/**
 * Kernel implementation templates.
 * 
 * Provided kernels include
 * - TSM2 kernel (also used with ISM2 proposed optimization 1)
 * - Alternative kernel for ISM2 proposed optimization 1
 * - Kernel for ISM2 proposed optimization 2
 */
#include "kernel_tsm2.cuh"
#include "kernel_ism2_opt1.cuh"
#include "kernel_ism2_opt2.cuh"

/**
 * Device-specific kernel selection.
 * Nvidia V100 only supported device right now.
 */
#include "v100/kernels_select.cuh"
