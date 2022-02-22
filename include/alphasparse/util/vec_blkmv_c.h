#pragma once

#include "../types.h"

#define BLOCK_DGEMV_ROW BLOCK_C_DGEMV_ROW
#define BLOCK_DGEMV_COL BLOCK_C_DGEMV_COL
#define BLOCK_DGEMV_ROW_CONJ2 BLOCK_C_DGEMV_ROW_CONJ2
#define BLOCK_DGEMV_COL_CONJ2 BLOCK_C_DGEMV_COL_CONJ2
//gemv c row
#ifdef CMLA
#define BLOCK_C_DGEMV_ROW(y, A, x, bs, lda)                                    \
    {                                                                          \
        float32x4_t prod00, prod10, prod20, prod30;                            \
        float32x4_t prod01, prod11, prod21, prod31;                            \
        float32x4_t a_v_00, a_v_10, a_v_20, a_v_30;                            \
        float32x4_t a_v_01, a_v_11, a_v_21, a_v_31;                            \
        float32x4_t x_v_0, x_v_1;                                              \
        float32x2_t y_v_0, y_v_1, y_v_2, y_v_3;                                \
        float32x2_t prod0, prod1, prod2, prod3;                                \
        float32x2_t x_v;                                                       \
        float32x2_t a_v_0, a_v_1, a_v_2, a_v_3;                                \
        float32x4_t zero = vdupq_n_f32(0);                                     \
        ALPHA_INT i = 0, j = 0;                                                  \
        for (i = 0; i < (bs)-3; i += 4)                                        \
        {                                                                      \
            ALPHA_Complex8 *Y = (y) + i;                                         \
            y_v_0 = vld1_f32((void *)Y);                                       \
            y_v_1 = vld1_f32((void *)(Y + 1));                                 \
            y_v_2 = vld1_f32((void *)(Y + 2));                                 \
            y_v_3 = vld1_f32((void *)(Y + 3));                                 \
            for (j = 0; j < (bs)-3; j += 4)                                    \
            {                                                                  \
                const ALPHA_Complex8 *X = (x) + j;                               \
                x_v_0 = vld1q_f32((void *)X);                                  \
                x_v_1 = vld1q_f32((void *)(X + 2));                            \
                const ALPHA_Complex8 *A00 = (A) + i * lda + j;                   \
                const ALPHA_Complex8 *A10 = (A) + (i + 1) * lda + j;             \
                const ALPHA_Complex8 *A20 = (A) + (i + 2) * lda + j;             \
                const ALPHA_Complex8 *A30 = (A) + (i + 3) * lda + j;             \
                const ALPHA_Complex8 *A01 = (A) + i * lda + j + 2;               \
                const ALPHA_Complex8 *A11 = (A) + (i + 1) * lda + j + 2;         \
                const ALPHA_Complex8 *A21 = (A) + (i + 2) * lda + j + 2;         \
                const ALPHA_Complex8 *A31 = (A) + (i + 3) * lda + j + 2;         \
                a_v_00 = vld1q_f32((void *)(A00));                             \
                a_v_10 = vld1q_f32((void *)(A10));                             \
                a_v_20 = vld1q_f32((void *)(A20));                             \
                a_v_30 = vld1q_f32((void *)(A30));                             \
                a_v_01 = vld1q_f32((void *)(A01));                             \
                a_v_11 = vld1q_f32((void *)(A11));                             \
                a_v_21 = vld1q_f32((void *)(A21));                             \
                a_v_31 = vld1q_f32((void *)(A31));                             \
                prod00 = vcmlaq_f32(zero, a_v_00, x_v_0);                      \
                prod10 = vcmlaq_f32(zero, a_v_10, x_v_0);                      \
                prod20 = vcmlaq_f32(zero, a_v_20, x_v_0);                      \
                prod30 = vcmlaq_f32(zero, a_v_30, x_v_0);                      \
                prod00 = vcmlaq_rot90_f32(prod00, a_v_00, x_v_0);              \
                prod10 = vcmlaq_rot90_f32(prod10, a_v_10, x_v_0);              \
                prod20 = vcmlaq_rot90_f32(prod20, a_v_20, x_v_0);              \
                prod30 = vcmlaq_rot90_f32(prod30, a_v_30, x_v_0);              \
                prod01 = vcmlaq_f32(prod00, a_v_01, x_v_1);                    \
                prod11 = vcmlaq_f32(prod10, a_v_11, x_v_1);                    \
                prod21 = vcmlaq_f32(prod20, a_v_21, x_v_1);                    \
                prod31 = vcmlaq_f32(prod30, a_v_31, x_v_1);                    \
                prod01 = vcmlaq_rot90_f32(prod01, a_v_01, x_v_1);              \
                prod11 = vcmlaq_rot90_f32(prod11, a_v_11, x_v_1);              \
                prod21 = vcmlaq_rot90_f32(prod21, a_v_21, x_v_1);              \
                prod31 = vcmlaq_rot90_f32(prod31, a_v_31, x_v_1);              \
                prod0 = vadd_f32(vget_low_f32(prod01), vget_high_f32(prod01)); \
                prod1 = vadd_f32(vget_low_f32(prod11), vget_high_f32(prod11)); \
                prod2 = vadd_f32(vget_low_f32(prod21), vget_high_f32(prod21)); \
                prod3 = vadd_f32(vget_low_f32(prod31), vget_high_f32(prod31)); \
                y_v_0 = vadd_f32((prod0), y_v_0);                              \
                y_v_1 = vadd_f32((prod1), y_v_1);                              \
                y_v_2 = vadd_f32((prod2), y_v_2);                              \
                y_v_3 = vadd_f32((prod3), y_v_3);                              \
            }                                                                  \
            for (; j < (bs); j++)                                              \
            {                                                                  \
                const ALPHA_Complex8 *X = (x) + j;                               \
                x_v = vld1_f32((void *)X);                                     \
                const ALPHA_Complex8 *A00 = (A) + i * lda + j;                   \
                const ALPHA_Complex8 *A10 = (A) + (i + 1) * lda + j;             \
                const ALPHA_Complex8 *A20 = (A) + (i + 2) * lda + j;             \
                const ALPHA_Complex8 *A30 = (A) + (i + 3) * lda + j;             \
                a_v_0 = vld1_f32((void *)(A00));                               \
                a_v_1 = vld1_f32((void *)(A10));                               \
                a_v_2 = vld1_f32((void *)(A20));                               \
                a_v_3 = vld1_f32((void *)(A30));                               \
                prod0 = vcmla_f32(y_v_0, a_v_0, x_v);                          \
                prod1 = vcmla_f32(y_v_1, a_v_1, x_v);                          \
                prod2 = vcmla_f32(y_v_2, a_v_2, x_v);                          \
                prod3 = vcmla_f32(y_v_3, a_v_3, x_v);                          \
                y_v_0 = vcmla_rot90_f32(prod0, a_v_0, x_v);                    \
                y_v_1 = vcmla_rot90_f32(prod1, a_v_1, x_v);                    \
                y_v_2 = vcmla_rot90_f32(prod2, a_v_2, x_v);                    \
                y_v_3 = vcmla_rot90_f32(prod3, a_v_3, x_v);                    \
            }                                                                  \
            vst1_f32((void *)(Y), y_v_0);                                      \
            vst1_f32((void *)(Y + 1), y_v_1);                                  \
            vst1_f32((void *)(Y + 2), y_v_2);                                  \
            vst1_f32((void *)(Y + 3), y_v_3);                                  \
        }                                                                      \
        for (; i < (bs); i++)                                                  \
        {                                                                      \
            ALPHA_Complex8 *Y = (y) + i;                                         \
            for (j = 0; j < bs; j++)                                           \
            {                                                                  \
                const ALPHA_Complex8 *_A = (A) + i * lda + j;                    \
                const ALPHA_Complex8 *X = (x) + j;                               \
                Y->real += _A->real * X->real - _A->imag * X->imag;            \
                Y->imag += _A->real * X->imag + _A->imag * X->real;            \
            }                                                                  \
        }                                                                      \
    }
#else
#define BLOCK_C_DGEMV_ROW(y, A, x, bs, lda)                     \
    for (ALPHA_INT i = 0; i < (bs); i++)                          \
    {                                                           \
        ALPHA_Complex8 *Y = (y) + i;                              \
        for (ALPHA_INT j = 0; j < bs; j++)                        \
        {                                                       \
            const ALPHA_Complex8 *_A = (A) + i * lda + j;         \
            const ALPHA_Complex8 *X = (x) + j;                    \
            Y->real += _A->real * X->real - _A->imag * X->imag; \
            Y->imag += _A->real * X->imag + _A->imag * X->real; \
        }                                                       \
    }
#endif
//gemv c col
#ifdef CMLA
#define BLOCK_C_DGEMV_COL(y, A, x, bs, lda)                                \
    {                                                                      \
        float32x4_t y_v_0, y_v_1;                                          \
        float32x4_t x_v_0, x_v_1, x_v_2, x_v_3;                            \
        float32x4_t a_v_00, a_v_01, a_v_02, a_v_03;                        \
        float32x4_t a_v_10, a_v_11, a_v_12, a_v_13;                        \
        ALPHA_INT i = 0, j = 0;                                              \
        for (j = 0; j < (bs)-3; j += 4)                                    \
        {                                                                  \
            const ALPHA_Complex8 *X = (x) + j;                               \
            x_v_0 = vreinterpretq_f32_f64(vld1q_dup_f64((void *)(X)));     \
            x_v_1 = vreinterpretq_f32_f64(vld1q_dup_f64((void *)(X + 1))); \
            x_v_2 = vreinterpretq_f32_f64(vld1q_dup_f64((void *)(X + 2))); \
            x_v_3 = vreinterpretq_f32_f64(vld1q_dup_f64((void *)(X + 3))); \
            for (i = 0; i < (bs)-3; i += 4)                                \
            {                                                              \
                ALPHA_Complex8 *Y = (y) + i;                                 \
                y_v_0 = vld1q_f32((void *)(Y));                            \
                y_v_1 = vld1q_f32((void *)(Y + 2));                        \
                const ALPHA_Complex8 *A00 = (A) + i + j * lda;               \
                const ALPHA_Complex8 *A01 = (A) + i + (j + 1) * lda;         \
                const ALPHA_Complex8 *A02 = (A) + i + (j + 2) * lda;         \
                const ALPHA_Complex8 *A03 = (A) + i + (j + 3) * lda;         \
                const ALPHA_Complex8 *A10 = (A) + i + 2 + j * lda;           \
                const ALPHA_Complex8 *A11 = (A) + i + 2 + (j + 1) * lda;     \
                const ALPHA_Complex8 *A12 = (A) + i + 2 + (j + 2) * lda;     \
                const ALPHA_Complex8 *A13 = (A) + i + 2 + (j + 3) * lda;     \
                a_v_00 = vld1q_f32((void *)(A00));                         \
                a_v_01 = vld1q_f32((void *)(A01));                         \
                a_v_02 = vld1q_f32((void *)(A02));                         \
                a_v_03 = vld1q_f32((void *)(A03));                         \
                a_v_10 = vld1q_f32((void *)(A10));                         \
                a_v_11 = vld1q_f32((void *)(A11));                         \
                a_v_12 = vld1q_f32((void *)(A12));                         \
                a_v_13 = vld1q_f32((void *)(A13));                         \
                y_v_0 = vcmlaq_f32(y_v_0, a_v_00, x_v_0);                  \
                y_v_0 = vcmlaq_f32(y_v_0, a_v_01, x_v_1);                  \
                y_v_0 = vcmlaq_f32(y_v_0, a_v_02, x_v_2);                  \
                y_v_0 = vcmlaq_f32(y_v_0, a_v_03, x_v_3);                  \
                y_v_1 = vcmlaq_f32(y_v_1, a_v_10, x_v_0);                  \
                y_v_1 = vcmlaq_f32(y_v_1, a_v_11, x_v_1);                  \
                y_v_1 = vcmlaq_f32(y_v_1, a_v_12, x_v_2);                  \
                y_v_1 = vcmlaq_f32(y_v_1, a_v_13, x_v_3);                  \
                y_v_0 = vcmlaq_rot90_f32(y_v_0, a_v_00, x_v_0);            \
                y_v_0 = vcmlaq_rot90_f32(y_v_0, a_v_01, x_v_1);            \
                y_v_0 = vcmlaq_rot90_f32(y_v_0, a_v_02, x_v_2);            \
                y_v_0 = vcmlaq_rot90_f32(y_v_0, a_v_03, x_v_3);            \
                y_v_1 = vcmlaq_rot90_f32(y_v_1, a_v_10, x_v_0);            \
                y_v_1 = vcmlaq_rot90_f32(y_v_1, a_v_11, x_v_1);            \
                y_v_1 = vcmlaq_rot90_f32(y_v_1, a_v_12, x_v_2);            \
                y_v_1 = vcmlaq_rot90_f32(y_v_1, a_v_13, x_v_3);            \
                vst1q_f32((void *)(Y), y_v_0);                             \
                vst1q_f32((void *)(Y + 2), y_v_1);                         \
            }                                                              \
            for (; i < (bs); i++)                                          \
            {                                                              \
                const ALPHA_Complex8 *A0 = (A) + i + j * lda;                \
                const ALPHA_Complex8 *A1 = (A) + i + (j + 1) * lda;          \
                const ALPHA_Complex8 *A2 = (A) + i + (j + 2) * lda;          \
                const ALPHA_Complex8 *A3 = (A) + i + (j + 3) * lda;          \
                ALPHA_Complex8 *Y = (y) + i;                                 \
                alpha_madde(Y[0], A0[0], X[0]);                              \
                alpha_madde(Y[0], A1[0], X[1]);                              \
                alpha_madde(Y[0], A2[0], X[2]);                              \
                alpha_madde(Y[0], A3[0], X[3]);                              \
            }                                                              \
        }                                                                  \
        for (; j < (bs); j++)                                              \
        {                                                                  \
            const ALPHA_Complex8 *X = (x) + j;                               \
            for (i = 0; i < (bs); i++)                                     \
            {                                                              \
                const ALPHA_Complex8 *A0 = (A) + i + j * lda;                \
                ALPHA_Complex8 *Y = (y) + i;                                 \
                alpha_madde(Y[0], A0[0], X[0]);                              \
            }                                                              \
        }                                                                  \
    }
#else
#define BLOCK_C_DGEMV_COL(y, A, x, bs, lda)               \
    {                                                     \
        for (ALPHA_INT j = 0; j < (bs); j++)                \
        {                                                 \
            const ALPHA_Number *X = (x) + j;                \
            for (ALPHA_INT i = 0; i < (bs); i++)            \
            {                                             \
                const ALPHA_Number *A0 = (A) + i + j * lda; \
                ALPHA_Number *Y = (y) + i;                  \
                alpha_madde(Y[0], A0[0], X[0]);             \
            }                                             \
        }                                                 \
    }
#endif

#ifdef CMLA
#define BLOCK_C_DGEMV_ROW_CONJ2(y, A, x, bs, lda)                              \
    {                                                                          \
        float32x4_t prod00, prod10, prod20, prod30;                            \
        float32x4_t prod01, prod11, prod21, prod31;                            \
        float32x4_t a_v_00, a_v_10, a_v_20, a_v_30;                            \
        float32x4_t a_v_01, a_v_11, a_v_21, a_v_31;                            \
        float32x4_t x_v_0, x_v_1;                                              \
        float32x2_t y_v_0, y_v_1, y_v_2, y_v_3;                                \
        float32x2_t prod0, prod1, prod2, prod3;                                \
        float32x2_t x_v;                                                       \
        float32x2_t a_v_0, a_v_1, a_v_2, a_v_3;                                \
        float32x4_t zero = vdupq_n_f32(0);                                     \
        ALPHA_INT i = 0, j = 0;                                                  \
        for (i = 0; i < (bs)-3; i += 4)                                        \
        {                                                                      \
            ALPHA_Complex8 *Y = (y) + i;                                         \
            y_v_0 = vld1_f32((void *)Y);                                       \
            y_v_1 = vld1_f32((void *)(Y + 1));                                 \
            y_v_2 = vld1_f32((void *)(Y + 2));                                 \
            y_v_3 = vld1_f32((void *)(Y + 3));                                 \
            for (j = 0; j < (bs)-3; j += 4)                                    \
            {                                                                  \
                const ALPHA_Complex8 *X = (x) + j;                               \
                x_v_0 = vld1q_f32((void *)X);                                  \
                x_v_1 = vld1q_f32((void *)(X + 2));                            \
                const ALPHA_Complex8 *A00 = (A) + i * lda + j;                   \
                const ALPHA_Complex8 *A10 = (A) + (i + 1) * lda + j;             \
                const ALPHA_Complex8 *A20 = (A) + (i + 2) * lda + j;             \
                const ALPHA_Complex8 *A30 = (A) + (i + 3) * lda + j;             \
                const ALPHA_Complex8 *A01 = (A) + i * lda + j + 2;               \
                const ALPHA_Complex8 *A11 = (A) + (i + 1) * lda + j + 2;         \
                const ALPHA_Complex8 *A21 = (A) + (i + 2) * lda + j + 2;         \
                const ALPHA_Complex8 *A31 = (A) + (i + 3) * lda + j + 2;         \
                a_v_00 = vld1q_f32((void *)(A00));                             \
                a_v_10 = vld1q_f32((void *)(A10));                             \
                a_v_20 = vld1q_f32((void *)(A20));                             \
                a_v_30 = vld1q_f32((void *)(A30));                             \
                a_v_01 = vld1q_f32((void *)(A01));                             \
                a_v_11 = vld1q_f32((void *)(A11));                             \
                a_v_21 = vld1q_f32((void *)(A21));                             \
                a_v_31 = vld1q_f32((void *)(A31));                             \
                prod00 = vcmlaq_f32(zero, a_v_00, x_v_0);                      \
                prod10 = vcmlaq_f32(zero, a_v_10, x_v_0);                      \
                prod20 = vcmlaq_f32(zero, a_v_20, x_v_0);                      \
                prod30 = vcmlaq_f32(zero, a_v_30, x_v_0);                      \
                prod00 = vcmlaq_rot270_f32(prod00, a_v_00, x_v_0);             \
                prod10 = vcmlaq_rot270_f32(prod10, a_v_10, x_v_0);             \
                prod20 = vcmlaq_rot270_f32(prod20, a_v_20, x_v_0);             \
                prod30 = vcmlaq_rot270_f32(prod30, a_v_30, x_v_0);             \
                prod01 = vcmlaq_f32(prod00, a_v_01, x_v_1);                    \
                prod11 = vcmlaq_f32(prod10, a_v_11, x_v_1);                    \
                prod21 = vcmlaq_f32(prod20, a_v_21, x_v_1);                    \
                prod31 = vcmlaq_f32(prod30, a_v_31, x_v_1);                    \
                prod01 = vcmlaq_rot270_f32(prod01, a_v_01, x_v_1);             \
                prod11 = vcmlaq_rot270_f32(prod11, a_v_11, x_v_1);             \
                prod21 = vcmlaq_rot270_f32(prod21, a_v_21, x_v_1);             \
                prod31 = vcmlaq_rot270_f32(prod31, a_v_31, x_v_1);             \
                prod0 = vadd_f32(vget_low_f32(prod01), vget_high_f32(prod01)); \
                prod1 = vadd_f32(vget_low_f32(prod11), vget_high_f32(prod11)); \
                prod2 = vadd_f32(vget_low_f32(prod21), vget_high_f32(prod21)); \
                prod3 = vadd_f32(vget_low_f32(prod31), vget_high_f32(prod31)); \
                y_v_0 = vadd_f32((prod0), y_v_0);                              \
                y_v_1 = vadd_f32((prod1), y_v_1);                              \
                y_v_2 = vadd_f32((prod2), y_v_2);                              \
                y_v_3 = vadd_f32((prod3), y_v_3);                              \
            }                                                                  \
            for (; j < (bs); j++)                                              \
            {                                                                  \
                const ALPHA_Complex8 *X = (x) + j;                               \
                x_v = vld1_f32((void *)X);                                     \
                const ALPHA_Complex8 *A00 = (A) + i * lda + j;                   \
                const ALPHA_Complex8 *A10 = (A) + (i + 1) * lda + j;             \
                const ALPHA_Complex8 *A20 = (A) + (i + 2) * lda + j;             \
                const ALPHA_Complex8 *A30 = (A) + (i + 3) * lda + j;             \
                a_v_0 = vld1_f32((void *)(A00));                               \
                a_v_1 = vld1_f32((void *)(A10));                               \
                a_v_2 = vld1_f32((void *)(A20));                               \
                a_v_3 = vld1_f32((void *)(A30));                               \
                prod0 = vcmla_f32(y_v_0, a_v_0, x_v);                          \
                prod1 = vcmla_f32(y_v_1, a_v_1, x_v);                          \
                prod2 = vcmla_f32(y_v_2, a_v_2, x_v);                          \
                prod3 = vcmla_f32(y_v_3, a_v_3, x_v);                          \
                y_v_0 = vcmla_rot270_f32(prod0, a_v_0, x_v);                   \
                y_v_1 = vcmla_rot270_f32(prod1, a_v_1, x_v);                   \
                y_v_2 = vcmla_rot270_f32(prod2, a_v_2, x_v);                   \
                y_v_3 = vcmla_rot270_f32(prod3, a_v_3, x_v);                   \
            }                                                                  \
            vst1_f32((void *)(Y), y_v_0);                                      \
            vst1_f32((void *)(Y + 1), y_v_1);                                  \
            vst1_f32((void *)(Y + 2), y_v_2);                                  \
            vst1_f32((void *)(Y + 3), y_v_3);                                  \
        }                                                                      \
        for (; i < (bs); i++)                                                  \
        {                                                                      \
            ALPHA_Complex8 *Y = (y) + i;                                         \
            for (j = 0; j < bs; j++)                                           \
            {                                                                  \
                const ALPHA_Complex8 *_A = (A) + i * lda + j;                    \
                const ALPHA_Complex8 *X = (x) + j;                               \
                Y->real += _A->real * X->real + _A->imag * X->imag;            \
                Y->imag += _A->real * X->imag - _A->imag * X->real;            \
            }                                                                  \
        }                                                                      \
    }
#else
#define BLOCK_C_DGEMV_ROW_CONJ2(y, A, x, bs, lda)               \
    for (ALPHA_INT i = 0; i < (bs); i++)                          \
    {                                                           \
        ALPHA_Complex8 *Y = (y) + i;                              \
        for (ALPHA_INT j = 0; j < bs; j++)                        \
        {                                                       \
            const ALPHA_Complex8 *_A = (A) + i * lda + j;         \
            const ALPHA_Complex8 *X = (x) + j;                    \
            Y->real += _A->real * X->real + _A->imag * X->imag; \
            Y->imag += _A->real * X->imag - _A->imag * X->real; \
        }                                                       \
    }
#endif
//gemv c col
#ifdef CMLA
#define BLOCK_C_DGEMV_COL_CONJ2(y, A, x, bs, lda)                          \
    {                                                                      \
        float32x4_t y_v_0, y_v_1;                                          \
        float32x4_t x_v_0, x_v_1, x_v_2, x_v_3;                            \
        float32x4_t a_v_00, a_v_01, a_v_02, a_v_03;                        \
        float32x4_t a_v_10, a_v_11, a_v_12, a_v_13;                        \
        ALPHA_INT i = 0, j = 0;                                              \
        for (j = 0; j < (bs)-3; j += 4)                                    \
        {                                                                  \
            const ALPHA_Complex8 *X = (x) + j;                               \
            x_v_0 = vreinterpretq_f32_f64(vld1q_dup_f64((void *)(X)));     \
            x_v_1 = vreinterpretq_f32_f64(vld1q_dup_f64((void *)(X + 1))); \
            x_v_2 = vreinterpretq_f32_f64(vld1q_dup_f64((void *)(X + 2))); \
            x_v_3 = vreinterpretq_f32_f64(vld1q_dup_f64((void *)(X + 3))); \
            for (i = 0; i < (bs)-3; i += 4)                                \
            {                                                              \
                ALPHA_Complex8 *Y = (y) + i;                                 \
                y_v_0 = vld1q_f32((void *)(Y));                            \
                y_v_1 = vld1q_f32((void *)(Y + 2));                        \
                const ALPHA_Complex8 *A00 = (A) + i + j * lda;               \
                const ALPHA_Complex8 *A01 = (A) + i + (j + 1) * lda;         \
                const ALPHA_Complex8 *A02 = (A) + i + (j + 2) * lda;         \
                const ALPHA_Complex8 *A03 = (A) + i + (j + 3) * lda;         \
                const ALPHA_Complex8 *A10 = (A) + i + 2 + j * lda;           \
                const ALPHA_Complex8 *A11 = (A) + i + 2 + (j + 1) * lda;     \
                const ALPHA_Complex8 *A12 = (A) + i + 2 + (j + 2) * lda;     \
                const ALPHA_Complex8 *A13 = (A) + i + 2 + (j + 3) * lda;     \
                a_v_00 = vld1q_f32((void *)(A00));                         \
                a_v_01 = vld1q_f32((void *)(A01));                         \
                a_v_02 = vld1q_f32((void *)(A02));                         \
                a_v_03 = vld1q_f32((void *)(A03));                         \
                a_v_10 = vld1q_f32((void *)(A10));                         \
                a_v_11 = vld1q_f32((void *)(A11));                         \
                a_v_12 = vld1q_f32((void *)(A12));                         \
                a_v_13 = vld1q_f32((void *)(A13));                         \
                y_v_0 = vcmlaq_f32(y_v_0, a_v_00, x_v_0);                  \
                y_v_0 = vcmlaq_f32(y_v_0, a_v_01, x_v_1);                  \
                y_v_0 = vcmlaq_f32(y_v_0, a_v_02, x_v_2);                  \
                y_v_0 = vcmlaq_f32(y_v_0, a_v_03, x_v_3);                  \
                y_v_1 = vcmlaq_f32(y_v_1, a_v_10, x_v_0);                  \
                y_v_1 = vcmlaq_f32(y_v_1, a_v_11, x_v_1);                  \
                y_v_1 = vcmlaq_f32(y_v_1, a_v_12, x_v_2);                  \
                y_v_1 = vcmlaq_f32(y_v_1, a_v_13, x_v_3);                  \
                y_v_0 = vcmlaq_rot270_f32(y_v_0, a_v_00, x_v_0);           \
                y_v_0 = vcmlaq_rot270_f32(y_v_0, a_v_01, x_v_1);           \
                y_v_0 = vcmlaq_rot270_f32(y_v_0, a_v_02, x_v_2);           \
                y_v_0 = vcmlaq_rot270_f32(y_v_0, a_v_03, x_v_3);           \
                y_v_1 = vcmlaq_rot270_f32(y_v_1, a_v_10, x_v_0);           \
                y_v_1 = vcmlaq_rot270_f32(y_v_1, a_v_11, x_v_1);           \
                y_v_1 = vcmlaq_rot270_f32(y_v_1, a_v_12, x_v_2);           \
                y_v_1 = vcmlaq_rot270_f32(y_v_1, a_v_13, x_v_3);           \
                vst1q_f32((void *)(Y), y_v_0);                             \
                vst1q_f32((void *)(Y + 2), y_v_1);                         \
            }                                                              \
            for (; i < (bs); i++)                                          \
            {                                                              \
                const ALPHA_Complex8 *A0 = (A) + i + j * lda;                \
                const ALPHA_Complex8 *A1 = (A) + i + (j + 1) * lda;          \
                const ALPHA_Complex8 *A2 = (A) + i + (j + 2) * lda;          \
                const ALPHA_Complex8 *A3 = (A) + i + (j + 3) * lda;          \
                ALPHA_Complex8 *Y = (y) + i;                                 \
                alpha_madde_2c(Y[0], A0[0], X[0]);                           \
                alpha_madde_2c(Y[0], A1[0], X[1]);                           \
                alpha_madde_2c(Y[0], A2[0], X[2]);                           \
                alpha_madde_2c(Y[0], A3[0], X[3]);                           \
            }                                                              \
        }                                                                  \
        for (; j < (bs); j++)                                              \
        {                                                                  \
            const ALPHA_Complex8 *X = (x) + j;                               \
            for (i = 0; i < (bs); i++)                                     \
            {                                                              \
                const ALPHA_Complex8 *A0 = (A) + i + j * lda;                \
                ALPHA_Complex8 *Y = (y) + i;                                 \
                alpha_madde_2c(Y[0], A0[0], X[0]);                           \
            }                                                              \
        }                                                                  \
    }
#else
#define BLOCK_C_DGEMV_COL_CONJ2(y, A, x, bs, lda)         \
    {                                                     \
        for (ALPHA_INT j = 0; j < (bs); j++)                \
        {                                                 \
            const ALPHA_Number *X = (x) + j;                \
            for (ALPHA_INT i = 0; i < (bs); i++)            \
            {                                             \
                const ALPHA_Number *A0 = (A) + i + j * lda; \
                ALPHA_Number *Y = (y) + i;                  \
                alpha_madde_2c(Y[0], A0[0], X[0]);          \
            }                                             \
        }                                                 \
    }
#endif
