#ifndef _COMMON_DCU_H
#define _COMMON_DCU_H

#include <hip/hip_runtime.h>
#include "alphasparse/types.h"
#include "alphasparse/spdef.h"

#ifdef S
#define wfreduce_sum64 wfreduce_s_sum64
#define wfreduce_sum wfreduce_s_sum
#define alpha_atomic_add atomic_s_add
#endif

#ifdef D
#define wfreduce_sum64 wfreduce_d_sum64
#define wfreduce_sum wfreduce_d_sum
#define alpha_atomic_add atomic_d_add
#endif

#ifdef C
#define wfreduce_sum64 wfreduce_c_sum64
#define wfreduce_sum wfreduce_c_sum
#define alpha_atomic_add atomic_c_add
#endif

#ifdef Z
#define wfreduce_sum64 wfreduce_z_sum64
#define wfreduce_sum wfreduce_z_sum
#define alpha_atomic_add atomic_z_add
#endif

// DPP-based float wavefront reduction sum
template <unsigned int WFSIZE>
__device__ __forceinline__ float wfreduce_s_sum(float sum)
{
    typedef union flt_b32
    {
        float val;
        uint32_t b32;
    } flt_b32_t;

    flt_b32_t upper_sum;
    flt_b32_t temp_sum;
    temp_sum.val = sum;

    if (WFSIZE > 1)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x111, 0xf, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if (WFSIZE > 2)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x112, 0xf, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if (WFSIZE > 4)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x114, 0xf, 0xe, false);
        temp_sum.val += upper_sum.val;
    }

    if (WFSIZE > 8)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x118, 0xf, 0xc, false);
        temp_sum.val += upper_sum.val;
    }

    if (WFSIZE > 16)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x142, 0xa, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if (WFSIZE > 32)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x143, 0xc, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    sum = temp_sum.val;
    return sum;
}

// DPP-based double wavefront reduction
template <unsigned int WFSIZE>
__device__ __forceinline__ double wfreduce_d_sum(double sum)
{
    typedef union dbl_b32
    {
        double val;
        uint32_t b32[2];
    } dbl_b32_t;

    dbl_b32_t upper_sum;
    dbl_b32_t temp_sum;
    temp_sum.val = sum;

    if (WFSIZE > 1)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x111, 0xf, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x111, 0xf, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if (WFSIZE > 2)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x112, 0xf, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x112, 0xf, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if (WFSIZE > 4)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x114, 0xf, 0xe, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x114, 0xf, 0xe, false);
        temp_sum.val += upper_sum.val;
    }

    if (WFSIZE > 8)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x118, 0xf, 0xc, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x118, 0xf, 0xc, false);
        temp_sum.val += upper_sum.val;
    }

    if (WFSIZE > 16)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x142, 0xa, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x142, 0xa, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if (WFSIZE > 32)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x143, 0xc, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x143, 0xc, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    sum = temp_sum.val;
    return sum;
}

// DPP-based complex float wavefront reduction sum
template <unsigned int WFSIZE>
__device__ __forceinline__ ALPHA_Complex8 wfreduce_c_sum(ALPHA_Complex8 sum)
{
    ALPHA_Complex8 res;
    res.real = wfreduce_s_sum<WFSIZE>(sum.real);
    res.imag = wfreduce_s_sum<WFSIZE>(sum.imag);
    return res;
}

// DPP-based complex double wavefront reduction
template <unsigned int WFSIZE>
__device__ __forceinline__ ALPHA_Complex16 wfreduce_z_sum(ALPHA_Complex16 sum)
{
    ALPHA_Complex16 res;
    res.real = wfreduce_d_sum<WFSIZE>(sum.real);
    res.imag = wfreduce_d_sum<WFSIZE>(sum.imag);
    return res;
}

// shuffle-based float wavefront reduction sum for wavefront size = 64
__device__ __forceinline__ float wfreduce_s_sum64(float sum)
{
    const int _WFSIZE = 64;
    for (int i = _WFSIZE >> 1; i > 0; i >>= 1)
    {
        sum += __shfl_xor(sum, i);
    }

    return sum;
}

// shuffle-based double wavefront reduction sum for wavefront size = 64
__device__ __forceinline__ double wfreduce_d_sum64(double sum)
{
    const int _WFSIZE = 64;
    for (int i = _WFSIZE >> 1; i > 0; i >>= 1)
    {
        sum += __shfl_xor(sum, i);
    }

    return sum;
}

// shuffle-based complex float wavefront reduction sum for wavefront size = 64
__device__ __forceinline__ ALPHA_Complex8 wfreduce_c_sum64(ALPHA_Complex8 sum)
{
    ALPHA_Complex8 tmp;
    tmp.real = wfreduce_s_sum64(sum.real);
    tmp.imag = wfreduce_s_sum64(sum.imag);
    return tmp;
}

// shuffle-based complex double wavefront reduction for wavefront size = 64
__device__ __forceinline__ ALPHA_Complex16 wfreduce_z_sum64(ALPHA_Complex16 sum)
{
    ALPHA_Complex16 tmp;
    tmp.real = wfreduce_d_sum64(sum.real);
    tmp.imag = wfreduce_d_sum64(sum.imag);
    return tmp;
}

// BSR indexing macros
#define BSR_IND(j, bi, bj, dir) ((dir == ALPHA_SPARSE_LAYOUT_ROW_MAJOR) ? BSR_IND_R(j, bi, bj) : BSR_IND_C(j, bi, bj))
#define BSR_IND_R(j, bi, bj) (bsr_dim * bsr_dim * (j) + (bi)*bsr_dim + (bj))
#define BSR_IND_C(j, bi, bj) (bsr_dim * bsr_dim * (j) + (bi) + (bj)*bsr_dim)

#define GEBSR_IND(j, bi, bj, dir) ((dir == ALPHA_SPARSE_LAYOUT_ROW_MAJOR) ? GEBSR_IND_R(j, bi, bj) : GEBSR_IND_C(j, bi, bj))
#define GEBSR_IND_R(j, bi, bj) (row_bsr_dim * col_bsr_dim * (j) + (bi)*col_bsr_dim + (bj))
#define GEBSR_IND_C(j, bi, bj) (row_bsr_dim * col_bsr_dim * (j) + (bi) + (bj)*row_bsr_dim)

// atomit add
#define atomic_s_add(a, b) (atomicAdd(&(a), (b)))

#define atomic_d_add(a, b) (atomicAdd(&(a), (b)))

#define atomic_c_add(a, b)                \
    {                                     \
        atomicAdd(&((a).real), (b).real); \
        atomicAdd(&((a).imag), (b).imag); \
    }

#define atomic_z_add(a, b)                \
    {                                     \
        atomicAdd(&((a).real), (b).real); \
        atomicAdd(&((a).imag), (b).imag); \
    }

#undef WFSIZE

#endif //_COMMON_DCU_H
