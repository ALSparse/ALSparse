#pragma once
#include "../types.h"
#include "../compute.h"

#define vec_fma2 vec_fma2_d
#define vec_fmai vec_fmai_d
#define VEC_FMA2 VEC_FMA2_D
#define VEC_FMS2 VEC_FMS2_D
#define VEC_MUL2 VEC_MUL2_D
#define VEC_MUL2_4 VEC_MUL2_D4
#define vec_fma2_tr vec_fma2_tr_d

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#ifdef __aarch64__
#define VEC_MUL2_D(y, x, __val, len)                                        \
    do                                                                      \
    {                                                                       \
        float64x2_t y_v_0, x_v_0, y_v_2, x_v_2, y_v_4, x_v_4, y_v_6, x_v_6; \
        float64x2_t val_v;                                                  \
        val_v = vdupq_n_f64((__val));                                       \
        int32_t __i = 0;                                                    \
        for (; __i < len - 7; __i += 8)                                     \
        {                                                                   \
            x_v_0 = vld1q_f64((double *)(x + __i));                         \
            x_v_2 = vld1q_f64((double *)(x + __i + 2));                     \
            x_v_4 = vld1q_f64((double *)(x + __i + 4));                     \
            x_v_6 = vld1q_f64((double *)(x + __i + 6));                     \
            y_v_0 = vmulq_f64(x_v_0, val_v);                                \
            y_v_2 = vmulq_f64(x_v_2, val_v);                                \
            y_v_4 = vmulq_f64(x_v_4, val_v);                                \
            y_v_6 = vmulq_f64(x_v_6, val_v);                                \
            vst1q_f64(&(y)[__i], y_v_0);                                    \
            vst1q_f64(&(y)[__i + 2], y_v_2);                                \
            vst1q_f64(&(y)[__i + 4], y_v_4);                                \
            vst1q_f64(&(y)[__i + 6], y_v_6);                                \
        }                                                                   \
        for (; __i < len - 3; __i += 4)                                     \
        {                                                                   \
            x_v_0 = vld1q_f64((double *)(x + __i));                         \
            x_v_2 = vld1q_f64((double *)(x + __i + 2));                     \
            y_v_0 = vmulq_f64(x_v_0, val_v);                                \
            y_v_2 = vmulq_f64(x_v_2, val_v);                                \
            vst1q_f64(&(y)[__i], y_v_0);                                    \
            vst1q_f64(&(y)[__i + 2], y_v_2);                                \
        }                                                                   \
        for (; __i < len; __i++)                                            \
        {                                                                   \
            (y)[__i] = (x)[__i] * __val;                                    \
        }                                                                   \
    } while (0)
#else
#define VEC_MUL2_D(y, x, __val, len)         \
    do                                       \
    {                                        \
        {                                    \
            int32_t __i = 0;                 \
            for (; __i < len; __i += 1)      \
            {                                \
                (y)[__i] = __val * (x)[__i]; \
            }                                \
        }                                    \
    } while (0)
#endif

#ifdef __aarch64__
#define VEC_MUL2_D4(y, x, __val)                          \
    {                                                     \
        float64x2_t val_v = vdupq_n_f64((__val));         \
        float64x2_t x_v_0 = vld1q_f64((double *)(x));     \
        float64x2_t x_v_2 = vld1q_f64((double *)(x + 2)); \
        float64x2_t y_v_0 = vmulq_f64(x_v_0, val_v);      \
        float64x2_t y_v_2 = vmulq_f64(x_v_2, val_v);      \
        vst1q_f64((y), y_v_0);                            \
        vst1q_f64((y) + 2, y_v_2);                        \
    }
#else
#define VEC_MUL2_D4(y, x, __val) VEC_MUL2_D(y, x, __val, 4)
#endif

#ifdef __aarch64__
#define VEC_FMS2_D(y, x, __val, len)                                        \
    do                                                                      \
    {                                                                       \
        float64x2_t y_v_0, x_v_0, y_v_2, x_v_2, y_v_4, x_v_4, y_v_6, x_v_6; \
        float64x2_t val_v;                                                  \
        val_v = vdupq_n_f64((__val));                                       \
        int32_t __i = 0;                                                    \
        for (; __i < len - 7; __i += 8)                                     \
        {                                                                   \
            y_v_0 = vld1q_f64((double *)(y + __i));                         \
            x_v_0 = vld1q_f64((double *)(x + __i));                         \
            y_v_2 = vld1q_f64((double *)(y + __i + 2));                     \
            x_v_2 = vld1q_f64((double *)(x + __i + 2));                     \
            y_v_4 = vld1q_f64((double *)(y + __i + 4));                     \
            x_v_4 = vld1q_f64((double *)(x + __i + 4));                     \
            y_v_6 = vld1q_f64((double *)(y + __i + 6));                     \
            x_v_6 = vld1q_f64((double *)(x + __i + 6));                     \
            y_v_0 = vfmsq_f64(y_v_0, x_v_0, val_v);                         \
            y_v_2 = vfmsq_f64(y_v_2, x_v_2, val_v);                         \
            y_v_4 = vfmsq_f64(y_v_4, x_v_4, val_v);                         \
            y_v_6 = vfmsq_f64(y_v_6, x_v_6, val_v);                         \
            vst1q_f64(&(y)[__i], y_v_0);                                    \
            vst1q_f64(&(y)[__i + 2], y_v_2);                                \
            vst1q_f64(&(y)[__i + 4], y_v_4);                                \
            vst1q_f64(&(y)[__i + 6], y_v_6);                                \
        }                                                                   \
        for (; __i < len - 3; __i += 4)                                     \
        {                                                                   \
            y_v_0 = vld1q_f64((double *)(y + __i));                         \
            x_v_0 = vld1q_f64((double *)(x + __i));                         \
            y_v_2 = vld1q_f64((double *)(y + __i + 2));                     \
            x_v_2 = vld1q_f64((double *)(x + __i + 2));                     \
            y_v_0 = vfmsq_f64(y_v_0, x_v_0, val_v);                         \
            y_v_2 = vfmsq_f64(y_v_2, x_v_2, val_v);                         \
            vst1q_f64(&(y)[__i], y_v_0);                                    \
            vst1q_f64(&(y)[__i + 2], y_v_2);                                \
        }                                                                   \
        for (; __i < len; __i++)                                            \
        {                                                                   \
            (y)[__i] -= (x)[__i] * __val;                                   \
        }                                                                   \
    } while (0)
#else
#define VEC_FMS2_D(y, x, __val, len)          \
    do                                        \
    {                                         \
        {                                     \
            int32_t __i = 0;                  \
            for (; __i < len; __i += 1)       \
            {                                 \
                (y)[__i] -= __val * (x)[__i]; \
            }                                 \
        }                                     \
    } while (0)
#endif
// y[] += x[] * val
#ifdef __aarch64__
#define VEC_FMA2_D(y, x, __val, len)                                        \
    do                                                                      \
    {                                                                       \
        float64x2_t y_v_0, x_v_0, y_v_2, x_v_2, y_v_4, x_v_4, y_v_6, x_v_6; \
        float64x2_t val_v;                                                  \
        val_v = vdupq_n_f64((__val));                                       \
        int32_t __i = 0;                                                    \
        for (; __i < len - 7; __i += 8)                                     \
        {                                                                   \
            y_v_0 = vld1q_f64((double *)(y + __i));                         \
            x_v_0 = vld1q_f64((double *)(x + __i));                         \
            y_v_2 = vld1q_f64((double *)(y + __i + 2));                     \
            x_v_2 = vld1q_f64((double *)(x + __i + 2));                     \
            y_v_4 = vld1q_f64((double *)(y + __i + 4));                     \
            x_v_4 = vld1q_f64((double *)(x + __i + 4));                     \
            y_v_6 = vld1q_f64((double *)(y + __i + 6));                     \
            x_v_6 = vld1q_f64((double *)(x + __i + 6));                     \
            y_v_0 = vfmaq_f64(y_v_0, x_v_0, val_v);                         \
            y_v_2 = vfmaq_f64(y_v_2, x_v_2, val_v);                         \
            y_v_4 = vfmaq_f64(y_v_4, x_v_4, val_v);                         \
            y_v_6 = vfmaq_f64(y_v_6, x_v_6, val_v);                         \
            vst1q_f64(&(y)[__i], y_v_0);                                    \
            vst1q_f64(&(y)[__i + 2], y_v_2);                                \
            vst1q_f64(&(y)[__i + 4], y_v_4);                                \
            vst1q_f64(&(y)[__i + 6], y_v_6);                                \
        }                                                                   \
        for (; __i < len - 3; __i += 4)                                     \
        {                                                                   \
            y_v_0 = vld1q_f64((double *)(y + __i));                         \
            x_v_0 = vld1q_f64((double *)(x + __i));                         \
            y_v_2 = vld1q_f64((double *)(y + __i + 2));                     \
            x_v_2 = vld1q_f64((double *)(x + __i + 2));                     \
            y_v_0 = vfmaq_f64(y_v_0, x_v_0, val_v);                         \
            y_v_2 = vfmaq_f64(y_v_2, x_v_2, val_v);                         \
            vst1q_f64(&(y)[__i], y_v_0);                                    \
            vst1q_f64(&(y)[__i + 2], y_v_2);                                \
        }                                                                   \
        for (; __i < len; __i++)                                            \
        {                                                                   \
            (y)[__i] += (x)[__i] * __val;                                   \
        }                                                                   \
    } while (0)
#else
#define VEC_FMA2_D(y, x, __val, len)          \
    do                                        \
    {                                         \
        {                                     \
            int32_t __i = 0;                  \
            for (; __i < len; __i += 1)       \
            {                                 \
                (y)[__i] += __val * (x)[__i]; \
            }                                 \
        }                                     \
    } while (0)
#endif

static inline void vec_fmai_d(double *y, const ALPHA_INT *indx, const double *x, const ALPHA_INT ns, const double val)
{
    ALPHA_INT ns4 = ((ns >> 2) << 2);
    ALPHA_INT i = 0;

    __asm__ volatile(
        "prfm pldl3strm, [%[x]]\n\t"
        "prfm pldl3strm, [%[indx]]\n\t"
        :
        : [ x ] "r"(x), [ indx ] "r"(indx));
    for (i = 0; i < ns4; i += 4)
    {
        __asm__ volatile(
            "prfm pldl3strm, [%[x]]\n\t"
            "prfm pldl3strm, [%[indx]]\n\t"
            :
            : [ x ] "r"(x + i + 4), [ indx ] "r"(indx + i + 4));

        real_madde(y[indx[i]], x[i], val);
        real_madde(y[indx[i + 1]], x[i + 1], val);
        real_madde(y[indx[i + 2]], x[i + 2], val);
        real_madde(y[indx[i + 3]], x[i + 3], val);
    }
    for (; i < ns; ++i)
    {
        real_madde(y[indx[i]], x[i], val);
    }
}
static inline void vec_fma2_d(double *y, const double *x, double val, ALPHA_INT len)
{
    ALPHA_INT i = 0;
#ifdef __aarch64__
    float64x2_t y_v_0, y_v_1, y_v_2, y_v_3, x_v_0, x_v_1, x_v_2, x_v_3;
    float64x2_t val_v;
    val_v = vdupq_n_f64(val);
    ALPHA_INT len8 = len - 7;
    for (; i < len8; i += 8)
    {
        y_v_0 = vld1q_f64(&y[i]);
        y_v_1 = vld1q_f64(&y[i + 2]);
        y_v_2 = vld1q_f64(&y[i + 4]);
        y_v_3 = vld1q_f64(&y[i + 6]);
        x_v_0 = vld1q_f64(&x[i]);
        x_v_1 = vld1q_f64(&x[i + 2]);
        x_v_2 = vld1q_f64(&x[i + 4]);
        x_v_3 = vld1q_f64(&x[i + 6]);
        y_v_0 = vfmaq_f64(y_v_0, x_v_0, val_v);
        y_v_1 = vfmaq_f64(y_v_1, x_v_1, val_v);
        y_v_2 = vfmaq_f64(y_v_2, x_v_2, val_v);
        y_v_3 = vfmaq_f64(y_v_3, x_v_3, val_v);
        vst1q_f64(&y[i], y_v_0);
        vst1q_f64(&y[i + 2], y_v_1);
        vst1q_f64(&y[i + 4], y_v_2);
        vst1q_f64(&y[i + 6], y_v_3);
    }
#endif
    for (; i < len; i += 1)
    {
        y[i] += val * x[i];
    }
}
#ifdef __aarch64__
static inline void vec_fma2_tr_d(double *y, const double *x, const double *val, const double diag, const double alpha, ALPHA_INT *col_indx, ALPHA_INT nnzr, ALPHA_INT out_y_col, ALPHA_INT r, ALPHA_INT ldy)
{
    ALPHA_INT ns4 = ((nnzr >> 2) << 2);
    ALPHA_INT i;

    float64x2_t t0, t1, t2, t3;
    float64x2_t y0, y1, y2, y3;
    float64x2_t v0, v1, v2, v3;
    t0 = vdupq_n_f64(0.0f);
    t1 = vdupq_n_f64(0.0f);
    t2 = vdupq_n_f64(0.0f);
    t3 = vdupq_n_f64(0.0f);

    for (i = 0; i < ns4; i += 4)
    {
        y0 = vld1q_f64(&y[col_indx[i] * ldy + out_y_col]); //ferch 4 values in y
        y1 = vld1q_f64(&y[col_indx[i + 1] * ldy + out_y_col]);
        y2 = vld1q_f64(&y[col_indx[i + 2] * ldy + out_y_col]);
        y3 = vld1q_f64(&y[col_indx[i + 3] * ldy + out_y_col]);
        v0 = vdupq_n_f64(val[i]);
        v1 = vdupq_n_f64(val[i + 1]);
        v2 = vdupq_n_f64(val[i + 2]);
        v3 = vdupq_n_f64(val[i + 3]);

        t0 = vfmaq_f64(t0, y0, v0);
        t1 = vfmaq_f64(t1, y1, v1);
        t2 = vfmaq_f64(t2, y2, v2);
        t3 = vfmaq_f64(t3, y3, v3);
    }

    for (; i < nnzr; ++i)
    {
        y0 = vld1q_f64(&y[col_indx[i] * ldy + out_y_col]);
        v0 = vdupq_n_f64(val[i]);
        t0 = vfmaq_f64(t0, y0, v0);
    }

    t0 = vaddq_f64(t0, t1);
    t2 = vaddq_f64(t2, t3);
    t0 = vaddq_f64(t0, t2);

    float64x2_t x0;
    x0 = vld1q_f64(x);
    x0 = vmulq_n_f64(x0, alpha);
    x0 = vsubq_f64(x0, t0);

    v0 = vdupq_n_f64(diag);

    y0 = vdivq_f64(x0, v0);
    vst1q_f64(&y[r * ldy + out_y_col], y0);
}
#endif