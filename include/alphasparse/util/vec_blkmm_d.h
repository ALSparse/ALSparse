#pragma once

#include "../types.h"
#define BLOCK_DGEMM_RR_SQUARE BLOCK_D_DGEMM_RR_SQUARE
#define BLOCK_DGEMM_RR_GENERAL BLOCK_D_DGEMM_RR_GENERAL
#define BLOCK_DGEMM_CR_GENERAL BLOCK_D_DGEMM_CR_GENERAL
#define iszero(num) (((num) * (num) < 1e-20))
//gemm square
#ifdef __aarch64__
#define BLOCK_D_DGEMM_RR_SQUARE(C, A, B, bs, ldc, lda, ldb)                             \
    for (ALPHA_INT __i = 0; __i < (bs); __i++)                                            \
    {                                                                                   \
        for (ALPHA_INT j = 0; j < (bs); j += 4)                                           \
        {                                                                               \
            float64x2_t z_v_0 = vld1q_f64((C) + __i * ldc + j);                         \
            float64x2_t z_v_1 = vld1q_f64((C) + __i * ldc + j + 2);                     \
            float64x2_t y_v_00, y_v_01, y_v_10, y_v_11, y_v_20, y_v_21, y_v_30, y_v_31; \
            float64x2_t x_v_0, x_v_1, x_v_2, x_v_3;                                     \
            for (ALPHA_INT k = 0; k < (bs); k += 4)                                       \
            {                                                                           \
                const double *_A = A + __i * (lda) + k;                                 \
                x_v_0 = vdupq_n_f64(_A[0]);                                             \
                x_v_1 = vdupq_n_f64(_A[1]);                                             \
                y_v_00 = vld1q_f64((B) + k * (ldb) + j);                                \
                y_v_01 = vld1q_f64((B) + k * (ldb) + j + 2);                            \
                y_v_10 = vld1q_f64((B) + (k + 1) * (ldb) + j);                          \
                y_v_11 = vld1q_f64((B) + (k + 1) * (ldb) + j + 2);                      \
                z_v_0 = vfmaq_f64(z_v_0, x_v_0, y_v_00);                                \
                z_v_0 = vfmaq_f64(z_v_0, x_v_1, y_v_10);                                \
                z_v_1 = vfmaq_f64(z_v_1, x_v_0, y_v_01);                                \
                z_v_1 = vfmaq_f64(z_v_1, x_v_1, y_v_11);                                \
                _A = A + __i * (lda) + (k + 2);                                         \
                x_v_2 = vdupq_n_f64(_A[0]);                                             \
                x_v_3 = vdupq_n_f64(_A[1]);                                             \
                y_v_20 = vld1q_f64((B) + (k + 2) * (ldb) + j);                          \
                y_v_21 = vld1q_f64((B) + (k + 2) * (ldb) + j + 2);                      \
                y_v_30 = vld1q_f64((B) + (k + 3) * (ldb) + j);                          \
                y_v_31 = vld1q_f64((B) + (k + 3) * (ldb) + j + 2);                      \
                z_v_0 = vfmaq_f64(z_v_0, x_v_2, y_v_20);                                \
                z_v_0 = vfmaq_f64(z_v_0, x_v_3, y_v_30);                                \
                z_v_1 = vfmaq_f64(z_v_1, x_v_2, y_v_21);                                \
                z_v_1 = vfmaq_f64(z_v_1, x_v_3, y_v_31);                                \
            }                                                                           \
            vst1q_f64((C) + __i * ldc + j, z_v_0);                                      \
            vst1q_f64((C) + __i * ldc + j + 2, z_v_1);                                  \
        }                                                                               \
    }
#else
#define BLOCK_D_DGEMM_RR_SQUARE(C, A, B, bs, ldc, lda, ldb)                            \
    for (ALPHA_INT __i = 0; __i < bs; __i++)                                             \
        for (ALPHA_INT j = 0; j < bs; j++)                                               \
        {                                                                              \
            ALPHA_INT m = 0;                                                             \
            for (; m < bs - 3; m += 4)                                                 \
            {                                                                          \
                (C)[__i * ldc + j] += (A)[__i * lda + m] * (B)[m * ldb + j];           \
                (C)[__i * ldc + j] += (A)[__i * lda + m + 1] * (B)[(m + 1) * ldb + j]; \
                (C)[__i * ldc + j] += (A)[__i * lda + m + 2] * (B)[(m + 2) * ldb + j]; \
                (C)[__i * ldc + j] += (A)[__i * lda + m + 3] * (B)[(m + 3) * ldb + j]; \
            }                                                                          \
        }
#endif
//gemm rr
#ifdef __aarch64__
#define BLOCK_D_DGEMM_RR_GENERAL(C, A, B, bs, ldc, lda, N, ldb) \
    {                                                           \
        float64x2_t z_v_0, z_v_1, z_v_2, z_v_3;                 \
        float64x2_t x_v_0;                                      \
        float64x2_t y_v_0, y_v_1, y_v_2, y_v_3;                 \
        for (ALPHA_INT __i = 0; __i < (bs); __i++)                \
        {                                                       \
            for (ALPHA_INT k = 0; k < (bs); k++)                  \
            {                                                   \
                const double *__A = (A) + __i * (lda) + k;      \
                x_v_0 = vdupq_n_f64(__A[0]);                    \
                ALPHA_INT j = 0;                                  \
                if (iszero(__A[0]))                             \
                    continue;                                   \
                for (; j < (N)-7; j += 8)                       \
                {                                               \
                    double *__C = (C) + __i * (ldc) + j;        \
                    const double *__B = (B) + k * (ldb) + j;    \
                    z_v_0 = vld1q_f64(__C + 0);                 \
                    z_v_1 = vld1q_f64(__C + 2);                 \
                    z_v_2 = vld1q_f64(__C + 4);                 \
                    z_v_3 = vld1q_f64(__C + 6);                 \
                    y_v_0 = vld1q_f64(__B + 0);                 \
                    y_v_1 = vld1q_f64(__B + 2);                 \
                    y_v_2 = vld1q_f64(__B + 4);                 \
                    y_v_3 = vld1q_f64(__B + 6);                 \
                    z_v_0 = vfmaq_f64(z_v_0, y_v_0, x_v_0);     \
                    z_v_1 = vfmaq_f64(z_v_1, y_v_1, x_v_0);     \
                    z_v_2 = vfmaq_f64(z_v_2, y_v_2, x_v_0);     \
                    z_v_3 = vfmaq_f64(z_v_3, y_v_3, x_v_0);     \
                    vst1q_f64(__C + 0, z_v_0);                  \
                    vst1q_f64(__C + 2, z_v_1);                  \
                    vst1q_f64(__C + 4, z_v_2);                  \
                    vst1q_f64(__C + 6, z_v_3);                  \
                }                                               \
                for (; j < (N)-1; j += 2)                       \
                {                                               \
                    double *__C = (C) + __i * (ldc) + j;        \
                    const double *__B = (B) + k * (ldb) + j;    \
                    z_v_0 = vld1q_f64(__C + 0);                 \
                    y_v_0 = vld1q_f64(__B + 0);                 \
                    z_v_0 = vfmaq_f64(z_v_0, y_v_0, x_v_0);     \
                    vst1q_f64(__C + 0, z_v_0);                  \
                }                                               \
                for (; j < (N); j += 1)                         \
                {                                               \
                    double *__C = (C) + __i * (ldc) + j;        \
                    const double *__B = (B) + k * (ldb) + j;    \
                    __C[0] += __A[0] * __B[0];                  \
                }                                               \
            }                                                   \
        }                                                       \
    }
#else
#define BLOCK_D_DGEMM_RR_GENERAL(C, A, B, bs, ldc, lda, N, ldb)              \
    for (ALPHA_INT __i = 0; __i < bs; __i++)                                   \
    {                                                                        \
        for (ALPHA_INT j = 0; j < bs; j++)                                     \
        {                                                                    \
            for (ALPHA_INT m = 0; m < bs; m++)                                 \
            {                                                                \
                (C)[__i * ldc + j] += (A)[__i * lda + m] * (B)[m * ldb + j]; \
            }                                                                \
        }                                                                    \
    }
#endif
//gemm cr
#ifdef __aarch64__
#define BLOCK_D_DGEMM_CR_GENERAL(C, A, B, bs, ldc, lda, N, ldb) \
    {                                                           \
        float64x2_t z_v_0, z_v_1, z_v_2, z_v_3;                 \
        float64x2_t x_v_0;                                      \
        float64x2_t y_v_0, y_v_1, y_v_2, y_v_3;                 \
        for (ALPHA_INT __i = 0; __i < (bs); __i++)                \
        {                                                       \
            for (ALPHA_INT k = 0; k < (bs); k++)                  \
            {                                                   \
                const double *__A = (A) + k * (lda) + __i;      \
                x_v_0 = vdupq_n_f64(__A[0]);                    \
                ALPHA_INT j = 0;                                  \
                if (iszero(__A[0]))                             \
                    continue;                                   \
                for (; j < (N)-7; j += 8)                       \
                {                                               \
                    double *__C = (C) + __i * (ldc) + j;        \
                    const double *__B = (B) + k * (ldb) + j;    \
                    z_v_0 = vld1q_f64(__C + 0);                 \
                    z_v_1 = vld1q_f64(__C + 2);                 \
                    z_v_2 = vld1q_f64(__C + 4);                 \
                    z_v_3 = vld1q_f64(__C + 6);                 \
                    y_v_0 = vld1q_f64(__B + 0);                 \
                    y_v_1 = vld1q_f64(__B + 2);                 \
                    y_v_2 = vld1q_f64(__B + 4);                 \
                    y_v_3 = vld1q_f64(__B + 6);                 \
                    z_v_0 = vfmaq_f64(z_v_0, y_v_0, x_v_0);     \
                    z_v_1 = vfmaq_f64(z_v_1, y_v_1, x_v_0);     \
                    z_v_2 = vfmaq_f64(z_v_2, y_v_2, x_v_0);     \
                    z_v_3 = vfmaq_f64(z_v_3, y_v_3, x_v_0);     \
                    vst1q_f64(__C + 0, z_v_0);                  \
                    vst1q_f64(__C + 2, z_v_1);                  \
                    vst1q_f64(__C + 4, z_v_2);                  \
                    vst1q_f64(__C + 6, z_v_3);                  \
                }                                               \
                for (; j < (N)-1; j += 2)                       \
                {                                               \
                    double *__C = (C) + __i * (ldc) + j;        \
                    const double *__B = (B) + k * (ldb) + j;    \
                    z_v_0 = vld1q_f64(__C + 0);                 \
                    y_v_0 = vld1q_f64(__B + 0);                 \
                    z_v_0 = vfmaq_f64(z_v_0, y_v_0, x_v_0);     \
                    vst1q_f64(__C + 0, z_v_0);                  \
                }                                               \
                for (; j < (N); j += 1)                         \
                {                                               \
                    double *__C = (C) + __i * (ldc) + j;        \
                    const double *__B = (B) + k * (ldb) + j;    \
                    __C[0] += __A[0] * __B[0];                  \
                }                                               \
            }                                                   \
        }                                                       \
    }
#else
#define BLOCK_D_DGEMM_CR_GENERAL(C, A, B, bs, ldc, lda, N, ldb)              \
    for (ALPHA_INT __i = 0; __i < bs; __i++)                                   \
    {                                                                        \
        for (ALPHA_INT j = 0; j < bs; j++)                                     \
        {                                                                    \
            for (ALPHA_INT m = 0; m < bs; m++)                                 \
            {                                                                \
                (C)[__i * ldc + j] += (A)[m * lda + __i] * (B)[m * ldb + j]; \
            }                                                                \
        }                                                                    \
    }
#endif
