#pragma once

#include "../types.h"
#define BLOCK_DGEMV_ROW BLOCK_S_DGEMV_ROW
#define BLOCK_DGEMV_COL BLOCK_S_DGEMV_COL
//gemv s row
#ifdef __aarch64__
#define BLOCK_S_DGEMV_ROW(y, A, x, bs, lda)                \
    {                                                      \
        float32x4_t prod0, prod1, prod2, prod3;            \
        float32x4_t a_v_0, a_v_1, a_v_2, a_v_3;            \
        float32x4_t x_v_0;                                 \
        ALPHA_INT i = 0, j = 0;                              \
        for (i = 0; i < (bs)-3; i += 4)                    \
        {                                                  \
            float *Y = (y) + i;                            \
            for (j = 0; j < (bs)-3; j += 4)                \
            {                                              \
                const float *X = (x) + j;                  \
                x_v_0 = vld1q_f32((void *)X);              \
                const float *A0 = (A) + i * lda + j;       \
                const float *A1 = (A) + (i + 1) * lda + j; \
                const float *A2 = (A) + (i + 2) * lda + j; \
                const float *A3 = (A) + (i + 3) * lda + j; \
                a_v_0 = vld1q_f32((void *)(A0));           \
                a_v_1 = vld1q_f32((void *)(A1));           \
                a_v_2 = vld1q_f32((void *)(A2));           \
                a_v_3 = vld1q_f32((void *)(A3));           \
                prod0 = vmulq_f32(a_v_0, x_v_0);           \
                prod1 = vmulq_f32(a_v_1, x_v_0);           \
                prod2 = vmulq_f32(a_v_2, x_v_0);           \
                prod3 = vmulq_f32(a_v_3, x_v_0);           \
                Y[0] += vaddvq_f32(prod0);                 \
                Y[1] += vaddvq_f32(prod1);                 \
                Y[2] += vaddvq_f32(prod2);                 \
                Y[3] += vaddvq_f32(prod3);                 \
            }                                              \
            for (; j < (bs); j++)                          \
            {                                              \
                const float *A0 = (A) + i * lda + j;       \
                const float *X = (x) + j;                  \
                const float *A1 = (A) + (i + 1) * lda + j; \
                const float *A2 = (A) + (i + 2) * lda + j; \
                const float *A3 = (A) + (i + 3) * lda + j; \
                Y[0] += A0[0] * X[0];                      \
                Y[1] += A1[0] * X[0];                      \
                Y[2] += A2[0] * X[0];                      \
                Y[3] += A3[0] * X[0];                      \
            }                                              \
        }                                                  \
        for (; i < (bs); i++)                              \
        {                                                  \
            for (j = 0; j < bs; j++)                       \
            {                                              \
                (y)[i] += (A)[i * lda + j] * (x)[j];       \
            }                                              \
        }                                                  \
    }
#else
#define BLOCK_S_DGEMV_ROW(y, A, x, bs, lda)          \
    for (ALPHA_INT __i = 0; __i < bs; __i++)           \
    {                                                \
        for (ALPHA_INT j = 0; j < bs; j++)             \
        {                                            \
            (y)[__i] += (A)[__i * lda + j] * (x)[j]; \
        }                                            \
    }
#endif
//gemv s col
#ifdef __aarch64__
#define BLOCK_S_DGEMV_COL(y, A, x, bs, lda)                                        \
    {                                                                              \
        float32x4_t y_v;                                                           \
        float32x4_t x_v_0, x_v_1, x_v_2, x_v_3;                                    \
        float32x4_t a_v_0, a_v_1, a_v_2, a_v_3;                                    \
        ALPHA_INT i = 0, j = 0;                                                      \
        for (j = 0; j < (bs)-3; j += 4)                                            \
        {                                                                          \
            const float *X = (x) + j;                                              \
            x_v_0 = vld1q_dup_f32((void *)(X));                                    \
            x_v_1 = vld1q_dup_f32((void *)(X + 1));                                \
            x_v_2 = vld1q_dup_f32((void *)(X + 2));                                \
            x_v_3 = vld1q_dup_f32((void *)(X + 3));                                \
            for (i = 0; i < (bs)-3; i += 4)                                        \
            {                                                                      \
                float *Y = (y) + i;                                                \
                y_v = vld1q_f32((void *)(Y));                                      \
                const float *A0 = (A) + i + j * lda;                               \
                const float *A1 = (A) + i + (j + 1) * lda;                         \
                const float *A2 = (A) + i + (j + 2) * lda;                         \
                const float *A3 = (A) + i + (j + 3) * lda;                         \
                a_v_0 = vld1q_f32((void *)(A0));                                   \
                a_v_1 = vld1q_f32((void *)(A1));                                   \
                a_v_2 = vld1q_f32((void *)(A2));                                   \
                a_v_3 = vld1q_f32((void *)(A3));                                   \
                y_v = vfmaq_f32(y_v, a_v_0, x_v_0);                                \
                y_v = vfmaq_f32(y_v, a_v_1, x_v_1);                                \
                y_v = vfmaq_f32(y_v, a_v_2, x_v_2);                                \
                y_v = vfmaq_f32(y_v, a_v_3, x_v_3);                                \
                vst1q_f32((void *)(Y), y_v);                                       \
            }                                                                      \
            for (; i < (bs); i++)                                                  \
            {                                                                      \
                const float *A0 = (A) + i + j * lda;                               \
                const float *A1 = (A) + i + (j + 1) * lda;                         \
                const float *A2 = (A) + i + (j + 2) * lda;                         \
                const float *A3 = (A) + i + (j + 3) * lda;                         \
                float *Y = (y) + i;                                                \
                Y[0] += A0[0] * X[0] + A1[0] * X[1] + A2[0] * X[2] + A3[0] * X[3]; \
            }                                                                      \
        }                                                                          \
        for (; j < (bs); j++)                                                      \
        {                                                                          \
            for (i = 0; i < (bs); i++)                                             \
            {                                                                      \
                (y)[i] += (A)[j * lda + i] * (x)[j];                               \
            }                                                                      \
        }                                                                          \
    }
#else
#define BLOCK_S_DGEMV_COL(y, A, x, bs, lda)              \
    for (ALPHA_INT j = 0; j < bs; j++)                     \
        for (ALPHA_INT __i = 0; __i < bs; __i++)           \
        {                                                \
            {                                            \
                (y)[__i] += (A)[j * lda + __i] * (x)[j]; \
            }                                            \
        }
#endif
