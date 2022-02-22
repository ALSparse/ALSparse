#pragma once

#include "../types.h"
#define BLOCK_DGEMV_ROW BLOCK_Z_DGEMV_ROW
#define BLOCK_DGEMV_COL BLOCK_Z_DGEMV_COL
#define BLOCK_DGEMV_ROW_CONJ2 BLOCK_Z_DGEMV_ROW_CONJ2
#define BLOCK_DGEMV_COL_CONJ2 BLOCK_Z_DGEMV_COL_CONJ2
//gemv z row
#ifdef CMLA
#define BLOCK_Z_DGEMV_ROW(y, A, x, bs, lda)                                                                                                                               \
    {                                                                                                                                                                     \
        float64x2_t y_v_00, y_v_10, y_v_20, y_v_30;                                                                                                                       \
        float64x2_t a_v_00, a_v_01, a_v_02, a_v_03;                                                                                                                       \
        float64x2_t a_v_10, a_v_11, a_v_12, a_v_13;                                                                                                                       \
        float64x2_t a_v_20, a_v_21, a_v_22, a_v_23;                                                                                                                       \
        float64x2_t a_v_30, a_v_31, a_v_32, a_v_33;                                                                                                                       \
        float64x2_t x_v_00, x_v_10, x_v_20, x_v_30;                                                                                                                       \
        ALPHA_INT i = 0, j = 0;                                                                                                                                             \
        for (i = 0; i < (bs)-3; i += 4)                                                                                                                                   \
        {                                                                                                                                                                 \
            ALPHA_Complex16 *Y = (y) + i;                                                                                                                                   \
            y_v_00 = vld1q_f64((void *)(Y + 0));                                                                                                                          \
            y_v_10 = vld1q_f64((void *)(Y + 1));                                                                                                                          \
            y_v_20 = vld1q_f64((void *)(Y + 2));                                                                                                                          \
            y_v_30 = vld1q_f64((void *)(Y + 3));                                                                                                                          \
            for (j = 0; j < (bs)-3; j += 4)                                                                                                                               \
            {                                                                                                                                                             \
                const ALPHA_Complex16 *X = (x) + j;                                                                                                                         \
                const ALPHA_Complex16 *_A00 = (A) + i * (lda) + j;                                                                                                          \
                const ALPHA_Complex16 *_A10 = (A) + (i + 1) * (lda) + j;                                                                                                    \
                const ALPHA_Complex16 *_A20 = (A) + (i + 2) * (lda) + j;                                                                                                    \
                const ALPHA_Complex16 *_A30 = (A) + (i + 3) * (lda) + j;                                                                                                    \
                a_v_00 = vld1q_f64((void *)_A00), a_v_01 = vld1q_f64((void *)(_A00 + 1)), a_v_02 = vld1q_f64((void *)(_A00 + 2)), a_v_03 = vld1q_f64((void *)(_A00 + 3)); \
                a_v_10 = vld1q_f64((void *)_A10), a_v_11 = vld1q_f64((void *)(_A10 + 1)), a_v_12 = vld1q_f64((void *)(_A10 + 2)), a_v_13 = vld1q_f64((void *)(_A10 + 3)); \
                a_v_20 = vld1q_f64((void *)_A20), a_v_21 = vld1q_f64((void *)(_A20 + 1)), a_v_22 = vld1q_f64((void *)(_A20 + 2)), a_v_23 = vld1q_f64((void *)(_A20 + 3)); \
                a_v_30 = vld1q_f64((void *)_A30), a_v_31 = vld1q_f64((void *)(_A30 + 1)), a_v_32 = vld1q_f64((void *)(_A30 + 2)), a_v_33 = vld1q_f64((void *)(_A30 + 3)); \
                x_v_00 = vld1q_f64((void *)(X + 0));                                                                                                                      \
                x_v_10 = vld1q_f64((void *)(X + 1));                                                                                                                      \
                x_v_20 = vld1q_f64((void *)(X + 2));                                                                                                                      \
                x_v_30 = vld1q_f64((void *)(X + 3));                                                                                                                      \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_00, x_v_00);                                                                                                              \
                y_v_10 = vcmlaq_f64(y_v_10, a_v_10, x_v_00);                                                                                                              \
                y_v_20 = vcmlaq_f64(y_v_20, a_v_20, x_v_00);                                                                                                              \
                y_v_30 = vcmlaq_f64(y_v_30, a_v_30, x_v_00);                                                                                                              \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_01, x_v_10);                                                                                                              \
                y_v_10 = vcmlaq_f64(y_v_10, a_v_11, x_v_10);                                                                                                              \
                y_v_20 = vcmlaq_f64(y_v_20, a_v_21, x_v_10);                                                                                                              \
                y_v_30 = vcmlaq_f64(y_v_30, a_v_31, x_v_10);                                                                                                              \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_02, x_v_20);                                                                                                              \
                y_v_10 = vcmlaq_f64(y_v_10, a_v_12, x_v_20);                                                                                                              \
                y_v_20 = vcmlaq_f64(y_v_20, a_v_22, x_v_20);                                                                                                              \
                y_v_30 = vcmlaq_f64(y_v_30, a_v_32, x_v_20);                                                                                                              \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_03, x_v_30);                                                                                                              \
                y_v_10 = vcmlaq_f64(y_v_10, a_v_13, x_v_30);                                                                                                              \
                y_v_20 = vcmlaq_f64(y_v_20, a_v_23, x_v_30);                                                                                                              \
                y_v_30 = vcmlaq_f64(y_v_30, a_v_33, x_v_30);                                                                                                              \
                y_v_00 = vcmlaq_rot90_f64(y_v_00, a_v_00, x_v_00);                                                                                                        \
                y_v_10 = vcmlaq_rot90_f64(y_v_10, a_v_10, x_v_00);                                                                                                        \
                y_v_20 = vcmlaq_rot90_f64(y_v_20, a_v_20, x_v_00);                                                                                                        \
                y_v_30 = vcmlaq_rot90_f64(y_v_30, a_v_30, x_v_00);                                                                                                        \
                y_v_00 = vcmlaq_rot90_f64(y_v_00, a_v_01, x_v_10);                                                                                                        \
                y_v_10 = vcmlaq_rot90_f64(y_v_10, a_v_11, x_v_10);                                                                                                        \
                y_v_20 = vcmlaq_rot90_f64(y_v_20, a_v_21, x_v_10);                                                                                                        \
                y_v_30 = vcmlaq_rot90_f64(y_v_30, a_v_31, x_v_10);                                                                                                        \
                y_v_00 = vcmlaq_rot90_f64(y_v_00, a_v_02, x_v_20);                                                                                                        \
                y_v_10 = vcmlaq_rot90_f64(y_v_10, a_v_12, x_v_20);                                                                                                        \
                y_v_20 = vcmlaq_rot90_f64(y_v_20, a_v_22, x_v_20);                                                                                                        \
                y_v_30 = vcmlaq_rot90_f64(y_v_30, a_v_32, x_v_20);                                                                                                        \
                y_v_00 = vcmlaq_rot90_f64(y_v_00, a_v_03, x_v_30);                                                                                                        \
                y_v_10 = vcmlaq_rot90_f64(y_v_10, a_v_13, x_v_30);                                                                                                        \
                y_v_20 = vcmlaq_rot90_f64(y_v_20, a_v_23, x_v_30);                                                                                                        \
                y_v_30 = vcmlaq_rot90_f64(y_v_30, a_v_33, x_v_30);                                                                                                        \
            }                                                                                                                                                             \
            for (; j < (bs); j++)                                                                                                                                         \
            {                                                                                                                                                             \
                const ALPHA_Complex16 *X = (x) + j;                                                                                                                         \
                const ALPHA_Complex16 *_A00 = (A) + i * (lda) + j;                                                                                                          \
                const ALPHA_Complex16 *_A10 = (A) + (i + 1) * (lda) + j;                                                                                                    \
                const ALPHA_Complex16 *_A20 = (A) + (i + 2) * (lda) + j;                                                                                                    \
                const ALPHA_Complex16 *_A30 = (A) + (i + 3) * (lda) + j;                                                                                                    \
                a_v_00 = vld1q_f64((void *)_A00);                                                                                                                         \
                a_v_10 = vld1q_f64((void *)_A10);                                                                                                                         \
                a_v_20 = vld1q_f64((void *)_A20);                                                                                                                         \
                a_v_30 = vld1q_f64((void *)_A30);                                                                                                                         \
                x_v_00 = vld1q_f64((void *)(X + 0));                                                                                                                      \
                x_v_10 = vld1q_f64((void *)(X + 1));                                                                                                                      \
                x_v_20 = vld1q_f64((void *)(X + 2));                                                                                                                      \
                x_v_30 = vld1q_f64((void *)(X + 3));                                                                                                                      \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_00, x_v_00);                                                                                                              \
                y_v_10 = vcmlaq_f64(y_v_10, a_v_10, x_v_00);                                                                                                              \
                y_v_20 = vcmlaq_f64(y_v_20, a_v_20, x_v_00);                                                                                                              \
                y_v_30 = vcmlaq_f64(y_v_30, a_v_30, x_v_00);                                                                                                              \
                y_v_00 = vcmlaq_rot90_f64(y_v_00, a_v_00, x_v_00);                                                                                                        \
                y_v_10 = vcmlaq_rot90_f64(y_v_10, a_v_10, x_v_00);                                                                                                        \
                y_v_20 = vcmlaq_rot90_f64(y_v_20, a_v_20, x_v_00);                                                                                                        \
                y_v_30 = vcmlaq_rot90_f64(y_v_30, a_v_30, x_v_00);                                                                                                        \
            }                                                                                                                                                             \
            vst1q_f64((void *)(Y + 0), y_v_00);                                                                                                                           \
            vst1q_f64((void *)(Y + 1), y_v_10);                                                                                                                           \
            vst1q_f64((void *)(Y + 2), y_v_20);                                                                                                                           \
            vst1q_f64((void *)(Y + 3), y_v_30);                                                                                                                           \
        }                                                                                                                                                                 \
        for (; i < (bs); i += 1)                                                                                                                                          \
        {                                                                                                                                                                 \
            ALPHA_Complex16 *Y = (y) + i;                                                                                                                                   \
            y_v_00 = vld1q_f64((void *)Y);                                                                                                                                \
            for (j = 0; j < (bs); j++)                                                                                                                                    \
            {                                                                                                                                                             \
                const ALPHA_Complex16 *X = (x) + j;                                                                                                                         \
                const ALPHA_Complex16 *_A00 = (A) + i * (lda) + j;                                                                                                          \
                a_v_00 = vld1q_f64((void *)_A00);                                                                                                                         \
                x_v_00 = vld1q_f64((void *)X);                                                                                                                            \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_00, x_v_00);                                                                                                              \
                y_v_00 = vcmlaq_rot90_f64(y_v_00, a_v_00, x_v_00);                                                                                                        \
            }                                                                                                                                                             \
            vst1q_f64((void *)Y, y_v_00);                                                                                                                                 \
        }                                                                                                                                                                 \
    }
#else
#define BLOCK_Z_DGEMV_ROW(y, A, x, bs, lda)                     \
    for (ALPHA_INT i = 0; i < (bs); i++)                          \
    {                                                           \
        ALPHA_Number *Y = (y) + i;                                \
        for (ALPHA_INT j = 0; j < bs; j++)                        \
        {                                                       \
            const ALPHA_Number *_A = (A) + i * lda + j;           \
            const ALPHA_Number *X = (x) + j;                      \
            Y->real += _A->real * X->real - _A->imag * X->imag; \
            Y->imag += _A->real * X->imag + _A->imag * X->real; \
        }                                                       \
    }
#endif
//gemv z col
#ifdef CMLA
#define BLOCK_Z_DGEMV_COL(y, A, x, bs, lda)                                                                                                                               \
    {                                                                                                                                                                     \
        float64x2_t y_v_00, y_v_10, y_v_20, y_v_30;                                                                                                                       \
        float64x2_t a_v_00, a_v_01, a_v_02, a_v_03;                                                                                                                       \
        float64x2_t a_v_10, a_v_11, a_v_12, a_v_13;                                                                                                                       \
        float64x2_t a_v_20, a_v_21, a_v_22, a_v_23;                                                                                                                       \
        float64x2_t a_v_30, a_v_31, a_v_32, a_v_33;                                                                                                                       \
        float64x2_t x_v_00, x_v_10, x_v_20, x_v_30;                                                                                                                       \
        ALPHA_INT i = 0, j = 0;                                                                                                                                             \
        for (j = 0; j < (bs)-3; j += 4)                                                                                                                                   \
        {                                                                                                                                                                 \
            const ALPHA_Complex16 *X = (x) + j;                                                                                                                             \
            x_v_00 = vld1q_f64((void *)((X)));                                                                                                                            \
            x_v_10 = vld1q_f64((void *)((X) + 1));                                                                                                                        \
            x_v_20 = vld1q_f64((void *)((X) + 2));                                                                                                                        \
            x_v_30 = vld1q_f64((void *)((X) + 3));                                                                                                                        \
            for (i = 0; i < (bs)-3; i += 4)                                                                                                                               \
            {                                                                                                                                                             \
                ALPHA_Complex16 *Y = (y) + i;                                                                                                                               \
                y_v_00 = vld1q_f64((void *)((Y)));                                                                                                                        \
                y_v_10 = vld1q_f64((void *)((Y) + 1));                                                                                                                    \
                y_v_20 = vld1q_f64((void *)((Y) + 2));                                                                                                                    \
                y_v_30 = vld1q_f64((void *)((Y) + 3));                                                                                                                    \
                const ALPHA_Complex16 *_A00 = (A) + j * (lda) + i;                                                                                                          \
                const ALPHA_Complex16 *_A01 = (A) + (j + 1) * (lda) + i;                                                                                                    \
                const ALPHA_Complex16 *_A02 = (A) + (j + 2) * (lda) + i;                                                                                                    \
                const ALPHA_Complex16 *_A03 = (A) + (j + 3) * (lda) + i;                                                                                                    \
                a_v_00 = vld1q_f64((void *)_A00), a_v_10 = vld1q_f64((void *)(_A00 + 1)), a_v_20 = vld1q_f64((void *)(_A00 + 2)), a_v_30 = vld1q_f64((void *)(_A00 + 3)); \
                a_v_01 = vld1q_f64((void *)_A01), a_v_11 = vld1q_f64((void *)(_A01 + 1)), a_v_21 = vld1q_f64((void *)(_A01 + 2)), a_v_31 = vld1q_f64((void *)(_A01 + 3)); \
                a_v_02 = vld1q_f64((void *)_A02), a_v_12 = vld1q_f64((void *)(_A02 + 1)), a_v_22 = vld1q_f64((void *)(_A02 + 2)), a_v_32 = vld1q_f64((void *)(_A02 + 3)); \
                a_v_03 = vld1q_f64((void *)_A03), a_v_13 = vld1q_f64((void *)(_A03 + 1)), a_v_23 = vld1q_f64((void *)(_A03 + 2)), a_v_33 = vld1q_f64((void *)(_A03 + 3)); \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_00, x_v_00);                                                                                                              \
                y_v_10 = vcmlaq_f64(y_v_10, a_v_10, x_v_00);                                                                                                              \
                y_v_20 = vcmlaq_f64(y_v_20, a_v_20, x_v_00);                                                                                                              \
                y_v_30 = vcmlaq_f64(y_v_30, a_v_30, x_v_00);                                                                                                              \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_01, x_v_10);                                                                                                              \
                y_v_10 = vcmlaq_f64(y_v_10, a_v_11, x_v_10);                                                                                                              \
                y_v_20 = vcmlaq_f64(y_v_20, a_v_21, x_v_10);                                                                                                              \
                y_v_30 = vcmlaq_f64(y_v_30, a_v_31, x_v_10);                                                                                                              \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_02, x_v_20);                                                                                                              \
                y_v_10 = vcmlaq_f64(y_v_10, a_v_12, x_v_20);                                                                                                              \
                y_v_20 = vcmlaq_f64(y_v_20, a_v_22, x_v_20);                                                                                                              \
                y_v_30 = vcmlaq_f64(y_v_30, a_v_32, x_v_20);                                                                                                              \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_03, x_v_30);                                                                                                              \
                y_v_10 = vcmlaq_f64(y_v_10, a_v_13, x_v_30);                                                                                                              \
                y_v_20 = vcmlaq_f64(y_v_20, a_v_23, x_v_30);                                                                                                              \
                y_v_30 = vcmlaq_f64(y_v_30, a_v_33, x_v_30);                                                                                                              \
                y_v_00 = vcmlaq_rot90_f64(y_v_00, a_v_00, x_v_00);                                                                                                        \
                y_v_10 = vcmlaq_rot90_f64(y_v_10, a_v_10, x_v_00);                                                                                                        \
                y_v_20 = vcmlaq_rot90_f64(y_v_20, a_v_20, x_v_00);                                                                                                        \
                y_v_30 = vcmlaq_rot90_f64(y_v_30, a_v_30, x_v_00);                                                                                                        \
                y_v_00 = vcmlaq_rot90_f64(y_v_00, a_v_01, x_v_10);                                                                                                        \
                y_v_10 = vcmlaq_rot90_f64(y_v_10, a_v_11, x_v_10);                                                                                                        \
                y_v_20 = vcmlaq_rot90_f64(y_v_20, a_v_21, x_v_10);                                                                                                        \
                y_v_30 = vcmlaq_rot90_f64(y_v_30, a_v_31, x_v_10);                                                                                                        \
                y_v_00 = vcmlaq_rot90_f64(y_v_00, a_v_02, x_v_20);                                                                                                        \
                y_v_10 = vcmlaq_rot90_f64(y_v_10, a_v_12, x_v_20);                                                                                                        \
                y_v_20 = vcmlaq_rot90_f64(y_v_20, a_v_22, x_v_20);                                                                                                        \
                y_v_30 = vcmlaq_rot90_f64(y_v_30, a_v_32, x_v_20);                                                                                                        \
                y_v_00 = vcmlaq_rot90_f64(y_v_00, a_v_03, x_v_30);                                                                                                        \
                y_v_10 = vcmlaq_rot90_f64(y_v_10, a_v_13, x_v_30);                                                                                                        \
                y_v_20 = vcmlaq_rot90_f64(y_v_20, a_v_23, x_v_30);                                                                                                        \
                y_v_30 = vcmlaq_rot90_f64(y_v_30, a_v_33, x_v_30);                                                                                                        \
                vst1q_f64((void *)((Y)), y_v_00);                                                                                                                         \
                vst1q_f64((void *)((Y) + 1), y_v_10);                                                                                                                     \
                vst1q_f64((void *)((Y) + 2), y_v_20);                                                                                                                     \
                vst1q_f64((void *)((Y) + 3), y_v_30);                                                                                                                     \
            }                                                                                                                                                             \
            for (; i < (bs); i += 1)                                                                                                                                      \
            {                                                                                                                                                             \
                ALPHA_Complex16 *Y = (y) + i;                                                                                                                               \
                y_v_00 = vld1q_f64((void *)((Y)));                                                                                                                        \
                                                                                                                                                                          \
                const ALPHA_Complex16 *_A00 = (A) + j * (lda) + i;                                                                                                          \
                const ALPHA_Complex16 *_A01 = (A) + (j + 1) * (lda) + i;                                                                                                    \
                const ALPHA_Complex16 *_A02 = (A) + (j + 2) * (lda) + i;                                                                                                    \
                const ALPHA_Complex16 *_A03 = (A) + (j + 3) * (lda) + i;                                                                                                    \
                a_v_00 = vld1q_f64((void *)_A00);                                                                                                                         \
                a_v_01 = vld1q_f64((void *)_A01);                                                                                                                         \
                a_v_02 = vld1q_f64((void *)_A02);                                                                                                                         \
                a_v_03 = vld1q_f64((void *)_A03);                                                                                                                         \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_00, x_v_00);                                                                                                              \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_01, x_v_10);                                                                                                              \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_02, x_v_20);                                                                                                              \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_03, x_v_30);                                                                                                              \
                y_v_00 = vcmlaq_rot90_f64(y_v_00, a_v_00, x_v_00);                                                                                                        \
                y_v_00 = vcmlaq_rot90_f64(y_v_00, a_v_01, x_v_10);                                                                                                        \
                y_v_00 = vcmlaq_rot90_f64(y_v_00, a_v_02, x_v_20);                                                                                                        \
                y_v_00 = vcmlaq_rot90_f64(y_v_00, a_v_03, x_v_30);                                                                                                        \
                vst1q_f64((void *)((Y)), y_v_00);                                                                                                                         \
            }                                                                                                                                                             \
        }                                                                                                                                                                 \
        for (; j < (bs); j++)                                                                                                                                             \
        {                                                                                                                                                                 \
            const ALPHA_Complex16 *X = (x) + j;                                                                                                                             \
            for (i = 0; i < (bs); i++)                                                                                                                                    \
            {                                                                                                                                                             \
                const ALPHA_Complex16 *A0 = (A) + i + j * lda;                                                                                                              \
                ALPHA_Complex16 *Y = (y) + i;                                                                                                                               \
                alpha_madde(Y[0], A0[0], X[0]);                                                                                                                             \
            }                                                                                                                                                             \
        }                                                                                                                                                                 \
    }
#else
#define BLOCK_Z_DGEMV_COL(y, A, x, bs, lda)               \
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

//gemv z row
#ifdef CMLA
#define BLOCK_Z_DGEMV_ROW_CONJ2(y, A, x, bs, lda)                                                                                                                         \
    {                                                                                                                                                                     \
        float64x2_t y_v_00, y_v_10, y_v_20, y_v_30;                                                                                                                       \
        float64x2_t a_v_00, a_v_01, a_v_02, a_v_03;                                                                                                                       \
        float64x2_t a_v_10, a_v_11, a_v_12, a_v_13;                                                                                                                       \
        float64x2_t a_v_20, a_v_21, a_v_22, a_v_23;                                                                                                                       \
        float64x2_t a_v_30, a_v_31, a_v_32, a_v_33;                                                                                                                       \
        float64x2_t x_v_00, x_v_10, x_v_20, x_v_30;                                                                                                                       \
        ALPHA_INT i = 0, j = 0;                                                                                                                                             \
        for (i = 0; i < (bs)-3; i += 4)                                                                                                                                   \
        {                                                                                                                                                                 \
            ALPHA_Complex16 *Y = (y) + i;                                                                                                                                   \
            y_v_00 = vld1q_f64((void *)(Y + 0));                                                                                                                          \
            y_v_10 = vld1q_f64((void *)(Y + 1));                                                                                                                          \
            y_v_20 = vld1q_f64((void *)(Y + 2));                                                                                                                          \
            y_v_30 = vld1q_f64((void *)(Y + 3));                                                                                                                          \
            for (j = 0; j < (bs)-3; j += 4)                                                                                                                               \
            {                                                                                                                                                             \
                const ALPHA_Complex16 *X = (x) + j;                                                                                                                         \
                const ALPHA_Complex16 *_A00 = (A) + i * (lda) + j;                                                                                                          \
                const ALPHA_Complex16 *_A10 = (A) + (i + 1) * (lda) + j;                                                                                                    \
                const ALPHA_Complex16 *_A20 = (A) + (i + 2) * (lda) + j;                                                                                                    \
                const ALPHA_Complex16 *_A30 = (A) + (i + 3) * (lda) + j;                                                                                                    \
                a_v_00 = vld1q_f64((void *)_A00), a_v_01 = vld1q_f64((void *)(_A00 + 1)), a_v_02 = vld1q_f64((void *)(_A00 + 2)), a_v_03 = vld1q_f64((void *)(_A00 + 3)); \
                a_v_10 = vld1q_f64((void *)_A10), a_v_11 = vld1q_f64((void *)(_A10 + 1)), a_v_12 = vld1q_f64((void *)(_A10 + 2)), a_v_13 = vld1q_f64((void *)(_A10 + 3)); \
                a_v_20 = vld1q_f64((void *)_A20), a_v_21 = vld1q_f64((void *)(_A20 + 1)), a_v_22 = vld1q_f64((void *)(_A20 + 2)), a_v_23 = vld1q_f64((void *)(_A20 + 3)); \
                a_v_30 = vld1q_f64((void *)_A30), a_v_31 = vld1q_f64((void *)(_A30 + 1)), a_v_32 = vld1q_f64((void *)(_A30 + 2)), a_v_33 = vld1q_f64((void *)(_A30 + 3)); \
                x_v_00 = vld1q_f64((void *)(X + 0));                                                                                                                      \
                x_v_10 = vld1q_f64((void *)(X + 1));                                                                                                                      \
                x_v_20 = vld1q_f64((void *)(X + 2));                                                                                                                      \
                x_v_30 = vld1q_f64((void *)(X + 3));                                                                                                                      \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_00, x_v_00);                                                                                                              \
                y_v_10 = vcmlaq_f64(y_v_10, a_v_10, x_v_00);                                                                                                              \
                y_v_20 = vcmlaq_f64(y_v_20, a_v_20, x_v_00);                                                                                                              \
                y_v_30 = vcmlaq_f64(y_v_30, a_v_30, x_v_00);                                                                                                              \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_01, x_v_10);                                                                                                              \
                y_v_10 = vcmlaq_f64(y_v_10, a_v_11, x_v_10);                                                                                                              \
                y_v_20 = vcmlaq_f64(y_v_20, a_v_21, x_v_10);                                                                                                              \
                y_v_30 = vcmlaq_f64(y_v_30, a_v_31, x_v_10);                                                                                                              \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_02, x_v_20);                                                                                                              \
                y_v_10 = vcmlaq_f64(y_v_10, a_v_12, x_v_20);                                                                                                              \
                y_v_20 = vcmlaq_f64(y_v_20, a_v_22, x_v_20);                                                                                                              \
                y_v_30 = vcmlaq_f64(y_v_30, a_v_32, x_v_20);                                                                                                              \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_03, x_v_30);                                                                                                              \
                y_v_10 = vcmlaq_f64(y_v_10, a_v_13, x_v_30);                                                                                                              \
                y_v_20 = vcmlaq_f64(y_v_20, a_v_23, x_v_30);                                                                                                              \
                y_v_30 = vcmlaq_f64(y_v_30, a_v_33, x_v_30);                                                                                                              \
                y_v_00 = vcmlaq_rot270_f64(y_v_00, a_v_00, x_v_00);                                                                                                        \
                y_v_10 = vcmlaq_rot270_f64(y_v_10, a_v_10, x_v_00);                                                                                                        \
                y_v_20 = vcmlaq_rot270_f64(y_v_20, a_v_20, x_v_00);                                                                                                        \
                y_v_30 = vcmlaq_rot270_f64(y_v_30, a_v_30, x_v_00);                                                                                                        \
                y_v_00 = vcmlaq_rot270_f64(y_v_00, a_v_01, x_v_10);                                                                                                        \
                y_v_10 = vcmlaq_rot270_f64(y_v_10, a_v_11, x_v_10);                                                                                                        \
                y_v_20 = vcmlaq_rot270_f64(y_v_20, a_v_21, x_v_10);                                                                                                        \
                y_v_30 = vcmlaq_rot270_f64(y_v_30, a_v_31, x_v_10);                                                                                                        \
                y_v_00 = vcmlaq_rot270_f64(y_v_00, a_v_02, x_v_20);                                                                                                        \
                y_v_10 = vcmlaq_rot270_f64(y_v_10, a_v_12, x_v_20);                                                                                                        \
                y_v_20 = vcmlaq_rot270_f64(y_v_20, a_v_22, x_v_20);                                                                                                        \
                y_v_30 = vcmlaq_rot270_f64(y_v_30, a_v_32, x_v_20);                                                                                                        \
                y_v_00 = vcmlaq_rot270_f64(y_v_00, a_v_03, x_v_30);                                                                                                        \
                y_v_10 = vcmlaq_rot270_f64(y_v_10, a_v_13, x_v_30);                                                                                                        \
                y_v_20 = vcmlaq_rot270_f64(y_v_20, a_v_23, x_v_30);                                                                                                        \
                y_v_30 = vcmlaq_rot270_f64(y_v_30, a_v_33, x_v_30);                                                                                                        \
            }                                                                                                                                                             \
            for (; j < (bs); j++)                                                                                                                                         \
            {                                                                                                                                                             \
                const ALPHA_Complex16 *X = (x) + j;                                                                                                                         \
                const ALPHA_Complex16 *_A00 = (A) + i * (lda) + j;                                                                                                          \
                const ALPHA_Complex16 *_A10 = (A) + (i + 1) * (lda) + j;                                                                                                    \
                const ALPHA_Complex16 *_A20 = (A) + (i + 2) * (lda) + j;                                                                                                    \
                const ALPHA_Complex16 *_A30 = (A) + (i + 3) * (lda) + j;                                                                                                    \
                a_v_00 = vld1q_f64((void *)_A00);                                                                                                                         \
                a_v_10 = vld1q_f64((void *)_A10);                                                                                                                         \
                a_v_20 = vld1q_f64((void *)_A20);                                                                                                                         \
                a_v_30 = vld1q_f64((void *)_A30);                                                                                                                         \
                x_v_00 = vld1q_f64((void *)(X + 0));                                                                                                                      \
                x_v_10 = vld1q_f64((void *)(X + 1));                                                                                                                      \
                x_v_20 = vld1q_f64((void *)(X + 2));                                                                                                                      \
                x_v_30 = vld1q_f64((void *)(X + 3));                                                                                                                      \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_00, x_v_00);                                                                                                              \
                y_v_10 = vcmlaq_f64(y_v_10, a_v_10, x_v_00);                                                                                                              \
                y_v_20 = vcmlaq_f64(y_v_20, a_v_20, x_v_00);                                                                                                              \
                y_v_30 = vcmlaq_f64(y_v_30, a_v_30, x_v_00);                                                                                                              \
                y_v_00 = vcmlaq_rot270_f64(y_v_00, a_v_00, x_v_00);                                                                                                        \
                y_v_10 = vcmlaq_rot270_f64(y_v_10, a_v_10, x_v_00);                                                                                                        \
                y_v_20 = vcmlaq_rot270_f64(y_v_20, a_v_20, x_v_00);                                                                                                        \
                y_v_30 = vcmlaq_rot270_f64(y_v_30, a_v_30, x_v_00);                                                                                                        \
            }                                                                                                                                                             \
            vst1q_f64((void *)(Y + 0), y_v_00);                                                                                                                           \
            vst1q_f64((void *)(Y + 1), y_v_10);                                                                                                                           \
            vst1q_f64((void *)(Y + 2), y_v_20);                                                                                                                           \
            vst1q_f64((void *)(Y + 3), y_v_30);                                                                                                                           \
        }                                                                                                                                                                 \
        for (; i < (bs); i += 1)                                                                                                                                          \
        {                                                                                                                                                                 \
            ALPHA_Complex16 *Y = (y) + i;                                                                                                                                   \
            y_v_00 = vld1q_f64((void *)Y);                                                                                                                                \
            for (j = 0; j < (bs); j++)                                                                                                                                    \
            {                                                                                                                                                             \
                const ALPHA_Complex16 *X = (x) + j;                                                                                                                         \
                const ALPHA_Complex16 *_A00 = (A) + i * (lda) + j;                                                                                                          \
                a_v_00 = vld1q_f64((void *)_A00);                                                                                                                         \
                x_v_00 = vld1q_f64((void *)X);                                                                                                                            \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_00, x_v_00);                                                                                                              \
                y_v_00 = vcmlaq_rot270_f64(y_v_00, a_v_00, x_v_00);                                                                                                        \
            }                                                                                                                                                             \
            vst1q_f64((void *)Y, y_v_00);                                                                                                                                 \
        }                                                                                                                                                                 \
    }
#else
#define BLOCK_Z_DGEMV_ROW_CONJ2(y, A, x, bs, lda)               \
    for (ALPHA_INT i = 0; i < (bs); i++)                          \
    {                                                           \
        ALPHA_Number *Y = (y) + i;                                \
        for (ALPHA_INT j = 0; j < bs; j++)                        \
        {                                                       \
            const ALPHA_Number *_A = (A) + i * lda + j;           \
            const ALPHA_Number *X = (x) + j;                      \
            Y->real += _A->real * X->real + _A->imag * X->imag; \
            Y->imag += _A->real * X->imag - _A->imag * X->real; \
        }                                                       \
    }
#endif
//gemv z col
#ifdef CMLA
#define BLOCK_Z_DGEMV_COL_CONJ2(y, A, x, bs, lda)                                                                                                                         \
    {                                                                                                                                                                     \
        float64x2_t y_v_00, y_v_10, y_v_20, y_v_30;                                                                                                                       \
        float64x2_t a_v_00, a_v_01, a_v_02, a_v_03;                                                                                                                       \
        float64x2_t a_v_10, a_v_11, a_v_12, a_v_13;                                                                                                                       \
        float64x2_t a_v_20, a_v_21, a_v_22, a_v_23;                                                                                                                       \
        float64x2_t a_v_30, a_v_31, a_v_32, a_v_33;                                                                                                                       \
        float64x2_t x_v_00, x_v_10, x_v_20, x_v_30;                                                                                                                       \
        ALPHA_INT i = 0, j = 0;                                                                                                                                             \
        for (j = 0; j < (bs)-3; j += 4)                                                                                                                                   \
        {                                                                                                                                                                 \
            const ALPHA_Complex16 *X = (x) + j;                                                                                                                             \
            x_v_00 = vld1q_f64((void *)((X)));                                                                                                                            \
            x_v_10 = vld1q_f64((void *)((X) + 1));                                                                                                                        \
            x_v_20 = vld1q_f64((void *)((X) + 2));                                                                                                                        \
            x_v_30 = vld1q_f64((void *)((X) + 3));                                                                                                                        \
            for (i = 0; i < (bs)-3; i += 4)                                                                                                                               \
            {                                                                                                                                                             \
                ALPHA_Complex16 *Y = (y) + i;                                                                                                                               \
                y_v_00 = vld1q_f64((void *)((Y)));                                                                                                                        \
                y_v_10 = vld1q_f64((void *)((Y) + 1));                                                                                                                    \
                y_v_20 = vld1q_f64((void *)((Y) + 2));                                                                                                                    \
                y_v_30 = vld1q_f64((void *)((Y) + 3));                                                                                                                    \
                const ALPHA_Complex16 *_A00 = (A) + j * (lda) + i;                                                                                                          \
                const ALPHA_Complex16 *_A01 = (A) + (j + 1) * (lda) + i;                                                                                                    \
                const ALPHA_Complex16 *_A02 = (A) + (j + 2) * (lda) + i;                                                                                                    \
                const ALPHA_Complex16 *_A03 = (A) + (j + 3) * (lda) + i;                                                                                                    \
                a_v_00 = vld1q_f64((void *)_A00), a_v_10 = vld1q_f64((void *)(_A00 + 1)), a_v_20 = vld1q_f64((void *)(_A00 + 2)), a_v_30 = vld1q_f64((void *)(_A00 + 3)); \
                a_v_01 = vld1q_f64((void *)_A01), a_v_11 = vld1q_f64((void *)(_A01 + 1)), a_v_21 = vld1q_f64((void *)(_A01 + 2)), a_v_31 = vld1q_f64((void *)(_A01 + 3)); \
                a_v_02 = vld1q_f64((void *)_A02), a_v_12 = vld1q_f64((void *)(_A02 + 1)), a_v_22 = vld1q_f64((void *)(_A02 + 2)), a_v_32 = vld1q_f64((void *)(_A02 + 3)); \
                a_v_03 = vld1q_f64((void *)_A03), a_v_13 = vld1q_f64((void *)(_A03 + 1)), a_v_23 = vld1q_f64((void *)(_A03 + 2)), a_v_33 = vld1q_f64((void *)(_A03 + 3)); \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_00, x_v_00);                                                                                                              \
                y_v_10 = vcmlaq_f64(y_v_10, a_v_10, x_v_00);                                                                                                              \
                y_v_20 = vcmlaq_f64(y_v_20, a_v_20, x_v_00);                                                                                                              \
                y_v_30 = vcmlaq_f64(y_v_30, a_v_30, x_v_00);                                                                                                              \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_01, x_v_10);                                                                                                              \
                y_v_10 = vcmlaq_f64(y_v_10, a_v_11, x_v_10);                                                                                                              \
                y_v_20 = vcmlaq_f64(y_v_20, a_v_21, x_v_10);                                                                                                              \
                y_v_30 = vcmlaq_f64(y_v_30, a_v_31, x_v_10);                                                                                                              \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_02, x_v_20);                                                                                                              \
                y_v_10 = vcmlaq_f64(y_v_10, a_v_12, x_v_20);                                                                                                              \
                y_v_20 = vcmlaq_f64(y_v_20, a_v_22, x_v_20);                                                                                                              \
                y_v_30 = vcmlaq_f64(y_v_30, a_v_32, x_v_20);                                                                                                              \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_03, x_v_30);                                                                                                              \
                y_v_10 = vcmlaq_f64(y_v_10, a_v_13, x_v_30);                                                                                                              \
                y_v_20 = vcmlaq_f64(y_v_20, a_v_23, x_v_30);                                                                                                              \
                y_v_30 = vcmlaq_f64(y_v_30, a_v_33, x_v_30);                                                                                                              \
                y_v_00 = vcmlaq_rot270_f64(y_v_00, a_v_00, x_v_00);                                                                                                        \
                y_v_10 = vcmlaq_rot270_f64(y_v_10, a_v_10, x_v_00);                                                                                                        \
                y_v_20 = vcmlaq_rot270_f64(y_v_20, a_v_20, x_v_00);                                                                                                        \
                y_v_30 = vcmlaq_rot270_f64(y_v_30, a_v_30, x_v_00);                                                                                                        \
                y_v_00 = vcmlaq_rot270_f64(y_v_00, a_v_01, x_v_10);                                                                                                        \
                y_v_10 = vcmlaq_rot270_f64(y_v_10, a_v_11, x_v_10);                                                                                                        \
                y_v_20 = vcmlaq_rot270_f64(y_v_20, a_v_21, x_v_10);                                                                                                        \
                y_v_30 = vcmlaq_rot270_f64(y_v_30, a_v_31, x_v_10);                                                                                                        \
                y_v_00 = vcmlaq_rot270_f64(y_v_00, a_v_02, x_v_20);                                                                                                        \
                y_v_10 = vcmlaq_rot270_f64(y_v_10, a_v_12, x_v_20);                                                                                                        \
                y_v_20 = vcmlaq_rot270_f64(y_v_20, a_v_22, x_v_20);                                                                                                        \
                y_v_30 = vcmlaq_rot270_f64(y_v_30, a_v_32, x_v_20);                                                                                                        \
                y_v_00 = vcmlaq_rot270_f64(y_v_00, a_v_03, x_v_30);                                                                                                        \
                y_v_10 = vcmlaq_rot270_f64(y_v_10, a_v_13, x_v_30);                                                                                                        \
                y_v_20 = vcmlaq_rot270_f64(y_v_20, a_v_23, x_v_30);                                                                                                        \
                y_v_30 = vcmlaq_rot270_f64(y_v_30, a_v_33, x_v_30);                                                                                                        \
                vst1q_f64((void *)((Y)), y_v_00);                                                                                                                         \
                vst1q_f64((void *)((Y) + 1), y_v_10);                                                                                                                     \
                vst1q_f64((void *)((Y) + 2), y_v_20);                                                                                                                     \
                vst1q_f64((void *)((Y) + 3), y_v_30);                                                                                                                     \
            }                                                                                                                                                             \
            for (; i < (bs); i += 1)                                                                                                                                      \
            {                                                                                                                                                             \
                ALPHA_Complex16 *Y = (y) + i;                                                                                                                               \
                y_v_00 = vld1q_f64((void *)((Y)));                                                                                                                        \
                                                                                                                                                                          \
                const ALPHA_Complex16 *_A00 = (A) + j * (lda) + i;                                                                                                          \
                const ALPHA_Complex16 *_A01 = (A) + (j + 1) * (lda) + i;                                                                                                    \
                const ALPHA_Complex16 *_A02 = (A) + (j + 2) * (lda) + i;                                                                                                    \
                const ALPHA_Complex16 *_A03 = (A) + (j + 3) * (lda) + i;                                                                                                    \
                a_v_00 = vld1q_f64((void *)_A00);                                                                                                                         \
                a_v_01 = vld1q_f64((void *)_A01);                                                                                                                         \
                a_v_02 = vld1q_f64((void *)_A02);                                                                                                                         \
                a_v_03 = vld1q_f64((void *)_A03);                                                                                                                         \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_00, x_v_00);                                                                                                              \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_01, x_v_10);                                                                                                              \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_02, x_v_20);                                                                                                              \
                y_v_00 = vcmlaq_f64(y_v_00, a_v_03, x_v_30);                                                                                                              \
                y_v_00 = vcmlaq_rot270_f64(y_v_00, a_v_00, x_v_00);                                                                                                        \
                y_v_00 = vcmlaq_rot270_f64(y_v_00, a_v_01, x_v_10);                                                                                                        \
                y_v_00 = vcmlaq_rot270_f64(y_v_00, a_v_02, x_v_20);                                                                                                        \
                y_v_00 = vcmlaq_rot270_f64(y_v_00, a_v_03, x_v_30);                                                                                                        \
                vst1q_f64((void *)((Y)), y_v_00);                                                                                                                         \
            }                                                                                                                                                             \
        }                                                                                                                                                                 \
        for (; j < (bs); j++)                                                                                                                                             \
        {                                                                                                                                                                 \
            const ALPHA_Complex16 *X = (x) + j;                                                                                                                             \
            for (i = 0; i < (bs); i++)                                                                                                                                    \
            {                                                                                                                                                             \
                const ALPHA_Complex16 *A0 = (A) + i + j * lda;                                                                                                              \
                ALPHA_Complex16 *Y = (y) + i;                                                                                                                               \
                alpha_madde(Y[0], A0[0], X[0]);                                                                                                                             \
            }                                                                                                                                                             \
        }                                                                                                                                                                 \
    }
#else
#define BLOCK_Z_DGEMV_COL_CONJ2(y, A, x, bs, lda)         \
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
