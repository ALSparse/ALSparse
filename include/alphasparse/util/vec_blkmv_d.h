#pragma once

#include "../types.h"
#define BLOCK_DGEMV_ROW BLOCK_D_DGEMV_ROW
#define BLOCK_DGEMV_COL BLOCK_D_DGEMV_COL
//gemv d row
#ifdef __aarch64__
#define BLOCK_D_DGEMV_ROW(y, A, x, bs, lda)                      \
    {                                                            \
        float64x2_t prod00, prod10, prod20, prod30;              \
        float64x2_t prod01, prod11, prod21, prod31;              \
        float64x2_t a_v_00, a_v_10, a_v_20, a_v_30;              \
        float64x2_t a_v_01, a_v_11, a_v_21, a_v_31;              \
        float64x2_t x_v_0, x_v_1;                                \
        float64x2_t zero = vdupq_n_f64(0);                       \
        ALPHA_INT i = 0, j = 0;                                    \
        for (i = 0; i < (bs)-3; i += 4)                          \
        {                                                        \
            double *Y = (y) + i;                                 \
            for (j = 0; j < (bs)-3; j += 4)                      \
            {                                                    \
                const double *X = (x) + j;                       \
                x_v_0 = vld1q_f64((void *)X);                    \
                x_v_1 = vld1q_f64((void *)(X + 2));              \
                const double *A00 = (A) + i * lda + j;           \
                const double *A10 = (A) + (i + 1) * lda + j;     \
                const double *A20 = (A) + (i + 2) * lda + j;     \
                const double *A30 = (A) + (i + 3) * lda + j;     \
                const double *A01 = (A) + i * lda + j + 2;       \
                const double *A11 = (A) + (i + 1) * lda + j + 2; \
                const double *A21 = (A) + (i + 2) * lda + j + 2; \
                const double *A31 = (A) + (i + 3) * lda + j + 2; \
                a_v_00 = vld1q_f64((void *)(A00));               \
                a_v_10 = vld1q_f64((void *)(A10));               \
                a_v_20 = vld1q_f64((void *)(A20));               \
                a_v_30 = vld1q_f64((void *)(A30));               \
                a_v_01 = vld1q_f64((void *)(A01));               \
                a_v_11 = vld1q_f64((void *)(A11));               \
                a_v_21 = vld1q_f64((void *)(A21));               \
                a_v_31 = vld1q_f64((void *)(A31));               \
                prod00 = vfmaq_f64(zero, a_v_00, x_v_0);         \
                prod10 = vfmaq_f64(zero, a_v_10, x_v_0);         \
                prod20 = vfmaq_f64(zero, a_v_20, x_v_0);         \
                prod30 = vfmaq_f64(zero, a_v_30, x_v_0);         \
                prod01 = vfmaq_f64(prod00, a_v_01, x_v_1);       \
                prod11 = vfmaq_f64(prod10, a_v_11, x_v_1);       \
                prod21 = vfmaq_f64(prod20, a_v_21, x_v_1);       \
                prod31 = vfmaq_f64(prod30, a_v_31, x_v_1);       \
                Y[0] += vaddvq_f64(prod01);                      \
                Y[1] += vaddvq_f64(prod11);                      \
                Y[2] += vaddvq_f64(prod21);                      \
                Y[3] += vaddvq_f64(prod31);                      \
            }                                                    \
            for (; j < (bs); j++)                                \
            {                                                    \
                const double *A0 = (A) + i * lda + j;            \
                const double *X = (x) + j;                       \
                const double *A1 = (A) + (i + 1) * lda + j;      \
                const double *A2 = (A) + (i + 2) * lda + j;      \
                const double *A3 = (A) + (i + 3) * lda + j;      \
                Y[0] += A0[0] * X[0];                            \
                Y[1] += A1[0] * X[0];                            \
                Y[2] += A2[0] * X[0];                            \
                Y[3] += A3[0] * X[0];                            \
            }                                                    \
        }                                                        \
        for (; i < (bs); i++)                                    \
        {                                                        \
            for (j = 0; j < bs; j++)                             \
            {                                                    \
                (y)[i] += (A)[i * lda + j] * (x)[j];             \
            }                                                    \
        }                                                        \
    }

#else
#define BLOCK_D_DGEMV_ROW(y, A, x, bs, lda)          \
    for (ALPHA_INT __i = 0; __i < bs; __i++)           \
    {                                                \
        for (ALPHA_INT j = 0; j < bs; j++)             \
        {                                            \
            (y)[__i] += (A)[__i * lda + j] * (x)[j]; \
        }                                            \
    }
#endif
//gemv d col
#ifdef __aarch64__
#define BLOCK_D_DGEMV_COL(y, A, x, bs, lda)                                        \
    {                                                                              \
        float64x2_t y_v_0, y_v_1;                                                  \
        float64x2_t x_v_0, x_v_1, x_v_2, x_v_3;                                    \
        float64x2_t a_v_00, a_v_01, a_v_02, a_v_03;                                \
        float64x2_t a_v_10, a_v_11, a_v_12, a_v_13;                                \
        ALPHA_INT i = 0, j = 0;                                                      \
        for (j = 0; j < (bs)-3; j += 4)                                            \
        {                                                                          \
            const double *X = (x) + j;                                             \
            x_v_0 = vld1q_dup_f64((void *)(X));                                    \
            x_v_1 = vld1q_dup_f64((void *)(X + 1));                                \
            x_v_2 = vld1q_dup_f64((void *)(X + 2));                                \
            x_v_3 = vld1q_dup_f64((void *)(X + 3));                                \
            for (i = 0; i < (bs)-3; i += 4)                                        \
            {                                                                      \
                double *Y = (y) + i;                                               \
                y_v_0 = vld1q_f64((void *)(Y));                                    \
                y_v_1 = vld1q_f64((void *)(Y + 2));                                \
                const double *A00 = (A) + i + j * lda;                             \
                const double *A01 = (A) + i + (j + 1) * lda;                       \
                const double *A02 = (A) + i + (j + 2) * lda;                       \
                const double *A03 = (A) + i + (j + 3) * lda;                       \
                const double *A10 = (A) + i + 2 + j * lda;                         \
                const double *A11 = (A) + i + 2 + (j + 1) * lda;                   \
                const double *A12 = (A) + i + 2 + (j + 2) * lda;                   \
                const double *A13 = (A) + i + 2 + (j + 3) * lda;                   \
                a_v_00 = vld1q_f64((void *)(A00));                                 \
                a_v_01 = vld1q_f64((void *)(A01));                                 \
                a_v_02 = vld1q_f64((void *)(A02));                                 \
                a_v_03 = vld1q_f64((void *)(A03));                                 \
                a_v_10 = vld1q_f64((void *)(A10));                                 \
                a_v_11 = vld1q_f64((void *)(A11));                                 \
                a_v_12 = vld1q_f64((void *)(A12));                                 \
                a_v_13 = vld1q_f64((void *)(A13));                                 \
                y_v_0 = vfmaq_f64(y_v_0, a_v_00, x_v_0);                           \
                y_v_0 = vfmaq_f64(y_v_0, a_v_01, x_v_1);                           \
                y_v_0 = vfmaq_f64(y_v_0, a_v_02, x_v_2);                           \
                y_v_0 = vfmaq_f64(y_v_0, a_v_03, x_v_3);                           \
                y_v_1 = vfmaq_f64(y_v_1, a_v_10, x_v_0);                           \
                y_v_1 = vfmaq_f64(y_v_1, a_v_11, x_v_1);                           \
                y_v_1 = vfmaq_f64(y_v_1, a_v_12, x_v_2);                           \
                y_v_1 = vfmaq_f64(y_v_1, a_v_13, x_v_3);                           \
                vst1q_f64((void *)(Y), y_v_0);                                     \
                vst1q_f64((void *)(Y + 2), y_v_1);                                 \
            }                                                                      \
            for (; i < (bs); i++)                                                  \
            {                                                                      \
                const double *A0 = (A) + i + j * lda;                              \
                const double *A1 = (A) + i + (j + 1) * lda;                        \
                const double *A2 = (A) + i + (j + 2) * lda;                        \
                const double *A3 = (A) + i + (j + 3) * lda;                        \
                double *Y = (y) + i;                                               \
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
#define BLOCK_D_DGEMV_ROW(y, A, x, bs, lda)          \
    for (ALPHA_INT __i = 0; __i < bs; __i++)           \
    {                                                \
        for (ALPHA_INT j = 0; j < bs; j++)             \
        {                                            \
            (y)[__i] += (A)[__i * lda + j] * (x)[j]; \
        }                                            \
    }
#endif
