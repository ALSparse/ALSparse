#pragma once

#include "../types.h"
#define BLOCK_DGEMM_RR_SQUARE BLOCK_S_DGEMM_RR_SQUARE
#define BLOCK_DGEMM_RR_GENERAL BLOCK_S_DGEMM_RR_GENERAL
#define BLOCK_DGEMM_CR_GENERAL BLOCK_S_DGEMM_CR_GENERAL
// #define iszero(num) (((num) > 0 && (num) < 1e-19) || (((num) < 0 && (num) > -1e-19)))
#define iszero(num) (((num) * (num) < 1e-20))
//gemm square
#ifdef __aarch64__
#define BLOCK_S_DGEMM_RR_SQUARE(C, A, B, bs, ldc, lda, ldb)                                                                                                                                   \
    {                                                                                                                                                                                         \
        float32x4_t z_v_00, z_v_10, z_v_20, z_v_30;                                                                                                                                           \
        float32x4_t x_v_00, x_v_01, x_v_02, x_v_03;                                                                                                                                           \
        float32x4_t x_v_10, x_v_11, x_v_12, x_v_13;                                                                                                                                           \
        float32x4_t x_v_20, x_v_21, x_v_22, x_v_23;                                                                                                                                           \
        float32x4_t x_v_30, x_v_31, x_v_32, x_v_33;                                                                                                                                           \
        float32x4_t y_v_00, y_v_10, y_v_20, y_v_30;                                                                                                                                           \
        for (ALPHA_INT __i = 0; __i < (bs); __i += 4)                                                                                                                                           \
        {                                                                                                                                                                                     \
            for (ALPHA_INT j = 0; j < (bs); j += 4)                                                                                                                                             \
            {                                                                                                                                                                                 \
                z_v_00 = vld1q_f32((C) + __i * (ldc) + j);                                                                                                                                    \
                z_v_10 = vld1q_f32((C) + (__i + 1) * (ldc) + j);                                                                                                                              \
                z_v_20 = vld1q_f32((C) + (__i + 2) * (ldc) + j);                                                                                                                              \
                z_v_30 = vld1q_f32((C) + (__i + 3) * (ldc) + j);                                                                                                                              \
                for (ALPHA_INT k = 0; k < (bs); k += 4)                                                                                                                                         \
                {                                                                                                                                                                             \
                    const float *_A00 = (A) + __i * (lda) + k;                                                                                                                                \
                    const float *_A10 = (A) + (__i + 1) * (lda) + k;                                                                                                                          \
                    const float *_A20 = (A) + (__i + 2) * (lda) + k;                                                                                                                          \
                    const float *_A30 = (A) + (__i + 3) * (lda) + k;                                                                                                                          \
                    x_v_00 = vld1q_dup_f32((void *)_A00), x_v_01 = vld1q_dup_f32((void *)(_A00 + 1)), x_v_02 = vld1q_dup_f32((void *)(_A00 + 2)), x_v_03 = vld1q_dup_f32((void *)(_A00 + 3)); \
                    x_v_10 = vld1q_dup_f32((void *)_A10), x_v_11 = vld1q_dup_f32((void *)(_A10 + 1)), x_v_12 = vld1q_dup_f32((void *)(_A10 + 2)), x_v_13 = vld1q_dup_f32((void *)(_A10 + 3)); \
                    x_v_20 = vld1q_dup_f32((void *)_A20), x_v_21 = vld1q_dup_f32((void *)(_A20 + 1)), x_v_22 = vld1q_dup_f32((void *)(_A20 + 2)), x_v_23 = vld1q_dup_f32((void *)(_A20 + 3)); \
                    x_v_30 = vld1q_dup_f32((void *)_A30), x_v_31 = vld1q_dup_f32((void *)(_A30 + 1)), x_v_32 = vld1q_dup_f32((void *)(_A30 + 2)), x_v_33 = vld1q_dup_f32((void *)(_A30 + 3)); \
                    y_v_00 = vld1q_f32((B) + k * (ldb) + j);                                                                                                                                  \
                    y_v_10 = vld1q_f32((B) + (k + 1) * (ldb) + j);                                                                                                                            \
                    y_v_20 = vld1q_f32((B) + (k + 2) * (ldb) + j);                                                                                                                            \
                    y_v_30 = vld1q_f32((B) + (k + 3) * (ldb) + j);                                                                                                                            \
                    z_v_00 = vfmaq_f32(z_v_00, x_v_00, y_v_00);                                                                                                                               \
                    z_v_10 = vfmaq_f32(z_v_10, x_v_10, y_v_00);                                                                                                                               \
                    z_v_20 = vfmaq_f32(z_v_20, x_v_20, y_v_00);                                                                                                                               \
                    z_v_30 = vfmaq_f32(z_v_30, x_v_30, y_v_00);                                                                                                                               \
                    z_v_00 = vfmaq_f32(z_v_00, x_v_01, y_v_10);                                                                                                                               \
                    z_v_10 = vfmaq_f32(z_v_10, x_v_11, y_v_10);                                                                                                                               \
                    z_v_20 = vfmaq_f32(z_v_20, x_v_21, y_v_10);                                                                                                                               \
                    z_v_30 = vfmaq_f32(z_v_30, x_v_31, y_v_10);                                                                                                                               \
                    z_v_00 = vfmaq_f32(z_v_00, x_v_02, y_v_20);                                                                                                                               \
                    z_v_10 = vfmaq_f32(z_v_10, x_v_12, y_v_20);                                                                                                                               \
                    z_v_20 = vfmaq_f32(z_v_20, x_v_22, y_v_20);                                                                                                                               \
                    z_v_30 = vfmaq_f32(z_v_30, x_v_32, y_v_20);                                                                                                                               \
                    z_v_00 = vfmaq_f32(z_v_00, x_v_03, y_v_30);                                                                                                                               \
                    z_v_10 = vfmaq_f32(z_v_10, x_v_13, y_v_30);                                                                                                                               \
                    z_v_20 = vfmaq_f32(z_v_20, x_v_23, y_v_30);                                                                                                                               \
                    z_v_30 = vfmaq_f32(z_v_30, x_v_33, y_v_30);                                                                                                                               \
                }                                                                                                                                                                             \
                vst1q_f32((C) + __i * (ldc) + j, z_v_00);                                                                                                                                     \
                vst1q_f32((C) + (__i + 1) * (ldc) + j, z_v_10);                                                                                                                               \
                vst1q_f32((C) + (__i + 2) * (ldc) + j, z_v_20);                                                                                                                               \
                vst1q_f32((C) + (__i + 3) * (ldc) + j, z_v_30);                                                                                                                               \
            }                                                                                                                                                                                 \
        }                                                                                                                                                                                     \
    }
#else
#define BLOCK_S_DGEMM_RR_SQUARE(C, A, B, bs, ldc, lda, ldb)                  \
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
//gemm rr
#ifdef __aarch64__
#define BLOCK_S_DGEMM_RR_GENERAL(C, A, B, bs, ldc, lda, N, ldb) \
    {                                                           \
        float32x4_t z_v_0, z_v_1, z_v_2, z_v_3;                 \
        float32x4_t x_v_0;                                      \
        float32x4_t y_v_0, y_v_1, y_v_2, y_v_3;                 \
        for (ALPHA_INT __i = 0; __i < (bs); __i++)                \
        {                                                       \
            for (ALPHA_INT k = 0; k < (bs); k++)                  \
            {                                                   \
                const float *__A = (A) + __i * (lda) + k;       \
                x_v_0 = vdupq_n_f32(__A[0]);                    \
                if (iszero(__A[0]))                             \
                    continue;                                   \
                ALPHA_INT j = 0;                                  \
                for (; j < (N)-15; j += 16)                     \
                {                                               \
                    float *__C = (C) + __i * (ldc) + j;         \
                    const float *__B = (B) + k * (ldb) + j;     \
                    z_v_0 = vld1q_f32(__C + 0);                 \
                    z_v_1 = vld1q_f32(__C + 4);                 \
                    z_v_2 = vld1q_f32(__C + 8);                 \
                    z_v_3 = vld1q_f32(__C + 12);                \
                    y_v_0 = vld1q_f32(__B + 0);                 \
                    y_v_1 = vld1q_f32(__B + 4);                 \
                    y_v_2 = vld1q_f32(__B + 8);                 \
                    y_v_3 = vld1q_f32(__B + 12);                \
                    z_v_0 = vfmaq_f32(z_v_0, y_v_0, x_v_0);     \
                    z_v_1 = vfmaq_f32(z_v_1, y_v_1, x_v_0);     \
                    z_v_2 = vfmaq_f32(z_v_2, y_v_2, x_v_0);     \
                    z_v_3 = vfmaq_f32(z_v_3, y_v_3, x_v_0);     \
                    vst1q_f32(__C + 0, z_v_0);                  \
                    vst1q_f32(__C + 4, z_v_1);                  \
                    vst1q_f32(__C + 8, z_v_2);                  \
                    vst1q_f32(__C + 12, z_v_3);                 \
                }                                               \
                for (; j < (N)-3; j += 4)                       \
                {                                               \
                    float *__C = (C) + __i * (ldc) + j;         \
                    const float *__B = (B) + k * (ldb) + j;     \
                    z_v_0 = vld1q_f32(__C + 0);                 \
                    y_v_0 = vld1q_f32(__B + 0);                 \
                    z_v_0 = vfmaq_f32(z_v_0, y_v_0, x_v_0);     \
                    vst1q_f32(__C + 0, z_v_0);                  \
                }                                               \
                for (; j < (N); j += 1)                         \
                {                                               \
                    float *__C = (C) + __i * (ldc) + j;         \
                    const float *__B = (B) + k * (ldb) + j;     \
                    __C[0] += __A[0] * __B[0];                  \
                }                                               \
            }                                                   \
        }                                                       \
    }
#else
#define BLOCK_S_DGEMM_RR_GENERAL(C, A, B, bs, ldc, lda, N, ldb)              \
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
#define BLOCK_S_DGEMM_CR_GENERAL(C, A, B, bs, ldc, lda, N, ldb) \
    {                                                           \
        float32x4_t z_v_0, z_v_1, z_v_2, z_v_3;                 \
        float32x4_t x_v_0;                                      \
        float32x4_t y_v_0, y_v_1, y_v_2, y_v_3;                 \
        for (ALPHA_INT __i = 0; __i < (bs); __i++)                \
        {                                                       \
            for (ALPHA_INT k = 0; k < (bs); k++)                  \
            {                                                   \
                const float *__A = (A) + k * (lda) + __i;       \
                x_v_0 = vdupq_n_f32(__A[0]);                    \
                if (iszero(__A[0]))                             \
                    continue;                                   \
                ALPHA_INT j = 0;                                  \
                for (; j < (N)-15; j += 16)                     \
                {                                               \
                    float *__C = (C) + __i * (ldc) + j;         \
                    const float *__B = (B) + k * (ldb) + j;     \
                    z_v_0 = vld1q_f32(__C + 0);                 \
                    z_v_1 = vld1q_f32(__C + 4);                 \
                    z_v_2 = vld1q_f32(__C + 8);                 \
                    z_v_3 = vld1q_f32(__C + 12);                \
                    y_v_0 = vld1q_f32(__B + 0);                 \
                    y_v_1 = vld1q_f32(__B + 4);                 \
                    y_v_2 = vld1q_f32(__B + 8);                 \
                    y_v_3 = vld1q_f32(__B + 12);                \
                    z_v_0 = vfmaq_f32(z_v_0, y_v_0, x_v_0);     \
                    z_v_1 = vfmaq_f32(z_v_1, y_v_1, x_v_0);     \
                    z_v_2 = vfmaq_f32(z_v_2, y_v_2, x_v_0);     \
                    z_v_3 = vfmaq_f32(z_v_3, y_v_3, x_v_0);     \
                    vst1q_f32(__C + 0, z_v_0);                  \
                    vst1q_f32(__C + 4, z_v_1);                  \
                    vst1q_f32(__C + 8, z_v_2);                  \
                    vst1q_f32(__C + 12, z_v_3);                 \
                }                                               \
                for (; j < (N)-3; j += 4)                       \
                {                                               \
                    float *__C = (C) + __i * (ldc) + j;         \
                    const float *__B = (B) + k * (ldb) + j;     \
                    z_v_0 = vld1q_f32(__C + 0);                 \
                    y_v_0 = vld1q_f32(__B + 0);                 \
                    z_v_0 = vfmaq_f32(z_v_0, y_v_0, x_v_0);     \
                    vst1q_f32(__C + 0, z_v_0);                  \
                }                                               \
                for (; j < (N); j += 1)                         \
                {                                               \
                    float *__C = (C) + __i * (ldc) + j;         \
                    const float *__B = (B) + k * (ldb) + j;     \
                    __C[0] += __A[0] * __B[0];                  \
                }                                               \
            }                                                   \
        }                                                       \
    }
#else
#define BLOCK_S_DGEMM_CR_GENERAL(C, A, B, bs, ldc, lda, N, ldb)              \
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
