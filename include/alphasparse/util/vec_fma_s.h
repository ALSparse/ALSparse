#pragma once

#include "../types.h"
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#include "../compute.h"

#define vec_fma2 vec_fma2_s
#define vec_fmai vec_fmai_s
#define VEC_FMA2 VEC_FMA2_S
#define VEC_FMS2 VEC_FMS2_S
#define VEC_MUL2 VEC_MUL2_S
#define VEC_MUL2_4 VEC_MUL2_S4
#define vec_fma2_tr vec_fma2_tr_s

#ifdef __aarch64__
#define VEC_MUL2_S4(y, x, __val)                     \
    {                                                \
        float32x4_t val_v = vdupq_n_f32((__val));    \
        float32x4_t x_v_0 = vld1q_f32((float *)(x)); \
        float32x4_t y_v_0 = vmulq_f32(x_v_0, val_v); \
        vst1q_f32((y), y_v_0);                       \
    }
#else
#define VEC_MUL2_S4(y, x, __val) VEC_MUL2_S(y, x, __val, 4)
#endif
#ifdef __aarch64__
#define VEC_MUL2_S(y, x, __val, len)                                        \
    {                                                                       \
        float32x4_t y_v_0, x_v_0, y_v_1, x_v_1, y_v_2, x_v_2, y_v_3, x_v_3; \
        float32x4_t val_v;                                                  \
        val_v = vdupq_n_f32((__val));                                       \
        int32_t len16 = (len)-15;                                           \
        int32_t __i = 0;                                                    \
        for (; __i < len16; __i += 16)                                      \
        {                                                                   \
            x_v_0 = vld1q_f32((float *)(x + __i));                          \
            x_v_1 = vld1q_f32((float *)(x + __i + 4));                      \
            x_v_2 = vld1q_f32((float *)(x + __i + 8));                      \
            x_v_3 = vld1q_f32((float *)(x + __i + 12));                     \
            y_v_0 = vmulq_f32(x_v_0, val_v);                                \
            y_v_1 = vmulq_f32(x_v_1, val_v);                                \
            y_v_2 = vmulq_f32(x_v_2, val_v);                                \
            y_v_3 = vmulq_f32(x_v_3, val_v);                                \
            vst1q_f32(&(y)[__i], y_v_0);                                    \
            vst1q_f32(&(y)[__i + 4], y_v_1);                                \
            vst1q_f32(&(y)[__i + 8], y_v_2);                                \
            vst1q_f32(&(y)[__i + 12], y_v_3);                               \
        }                                                                   \
        for (; __i < len - 3; __i += 4)                                     \
        {                                                                   \
            x_v_0 = vld1q_f32((float *)(x + __i));                          \
            y_v_0 = vmulq_f32(x_v_0, val_v);                                \
            vst1q_f32(&(y)[__i], y_v_0);                                    \
        }                                                                   \
        for (; __i < len; __i++)                                            \
        {                                                                   \
            (y)[__i] = (x)[__i] * __val;                                    \
        }                                                                   \
    }
#else
#define VEC_MUL2_S(y, x, __val, len)         \
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
#define VEC_FMS2_S(y, x, __val, len)                                        \
    do                                                                      \
    {                                                                       \
        float32x4_t y_v_0, x_v_0, y_v_1, x_v_1, y_v_2, x_v_2, y_v_3, x_v_3; \
        float32x4_t val_v;                                                  \
        val_v = vdupq_n_f32((__val));                                       \
        int32_t len16 = (len)-15;                                           \
        int32_t __i = 0;                                                    \
        for (; __i < len16; __i += 16)                                      \
        {                                                                   \
            y_v_0 = vld1q_f32((float *)(y + __i));                          \
            x_v_0 = vld1q_f32((float *)(x + __i));                          \
            y_v_1 = vld1q_f32((float *)(y + __i + 4));                      \
            x_v_1 = vld1q_f32((float *)(x + __i + 4));                      \
            y_v_2 = vld1q_f32((float *)(y + __i + 8));                      \
            x_v_2 = vld1q_f32((float *)(x + __i + 8));                      \
            y_v_3 = vld1q_f32((float *)(y + __i + 12));                     \
            x_v_3 = vld1q_f32((float *)(x + __i + 12));                     \
            y_v_0 = vfmsq_f32(y_v_0, x_v_0, val_v);                         \
            y_v_1 = vfmsq_f32(y_v_1, x_v_1, val_v);                         \
            y_v_2 = vfmsq_f32(y_v_2, x_v_2, val_v);                         \
            y_v_3 = vfmsq_f32(y_v_3, x_v_3, val_v);                         \
            vst1q_f32(&(y)[__i], y_v_0);                                    \
            vst1q_f32(&(y)[__i + 4], y_v_1);                                \
            vst1q_f32(&(y)[__i + 8], y_v_2);                                \
            vst1q_f32(&(y)[__i + 12], y_v_3);                               \
        }                                                                   \
        for (; __i < len - 3; __i += 4)                                     \
        {                                                                   \
            y_v_0 = vld1q_f32((float *)(y + __i));                          \
            x_v_0 = vld1q_f32((float *)(x + __i));                          \
            y_v_0 = vfmsq_f32(y_v_0, x_v_0, val_v);                         \
            vst1q_f32(&(y)[__i], y_v_0);                                    \
        }                                                                   \
        for (; __i < len; __i++)                                            \
        {                                                                   \
            (y)[__i] -= (x)[__i] * __val;                                   \
        }                                                                   \
    } while (0)
#else
#define VEC_FMS2_S(y, x, __val, len)          \
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
#ifdef __aarch64__
#define VEC_FMA2_S(y, x, __val, len)                                        \
    do                                                                      \
    {                                                                       \
        float32x4_t y_v_0, x_v_0, y_v_1, x_v_1, y_v_2, x_v_2, y_v_3, x_v_3; \
        float32x4_t val_v;                                                  \
        val_v = vdupq_n_f32((__val));                                       \
        int32_t len16 = (len)-15;                                           \
        int32_t __i = 0;                                                    \
        for (; __i < len16; __i += 16)                                      \
        {                                                                   \
            y_v_0 = vld1q_f32((float *)(y + __i));                          \
            x_v_0 = vld1q_f32((float *)(x + __i));                          \
            y_v_1 = vld1q_f32((float *)(y + __i + 4));                      \
            x_v_1 = vld1q_f32((float *)(x + __i + 4));                      \
            y_v_2 = vld1q_f32((float *)(y + __i + 8));                      \
            x_v_2 = vld1q_f32((float *)(x + __i + 8));                      \
            y_v_3 = vld1q_f32((float *)(y + __i + 12));                     \
            x_v_3 = vld1q_f32((float *)(x + __i + 12));                     \
            y_v_0 = vfmaq_f32(y_v_0, x_v_0, val_v);                         \
            y_v_1 = vfmaq_f32(y_v_1, x_v_1, val_v);                         \
            y_v_2 = vfmaq_f32(y_v_2, x_v_2, val_v);                         \
            y_v_3 = vfmaq_f32(y_v_3, x_v_3, val_v);                         \
            vst1q_f32(&(y)[__i], y_v_0);                                    \
            vst1q_f32(&(y)[__i + 4], y_v_1);                                \
            vst1q_f32(&(y)[__i + 8], y_v_2);                                \
            vst1q_f32(&(y)[__i + 12], y_v_3);                               \
        }                                                                   \
        for (; __i < len - 3; __i += 4)                                     \
        {                                                                   \
            y_v_0 = vld1q_f32((float *)(y + __i));                          \
            x_v_0 = vld1q_f32((float *)(x + __i));                          \
            y_v_0 = vfmaq_f32(y_v_0, x_v_0, val_v);                         \
            vst1q_f32(&(y)[__i], y_v_0);                                    \
        }                                                                   \
        for (; __i < len; __i++)                                            \
        {                                                                   \
            (y)[__i] += (x)[__i] * __val;                                   \
        }                                                                   \
    } while (0)
#else
#define VEC_FMA2_S(y, x, __val, len)          \
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

static inline void vec_fmai_s(float *y, const ALPHA_INT *indx, const float *x, const ALPHA_INT ns, const float val)
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
static inline void vec_fma2_s(float *y, const float *x, float val, ALPHA_INT len)
{
    ALPHA_INT i = 0;
#ifdef __aarch64__
    float32x4_t y_v_0, y_v_1, y_v_2, y_v_3, x_v_0, x_v_1, x_v_2, x_v_3;
    float32x4_t val_v;
    val_v = vdupq_n_f32(val);
    ALPHA_INT len16 = len - 15;
    for (; i < len16; i += 16)
    {
        y_v_0 = vld1q_f32(&y[i]);
        y_v_1 = vld1q_f32(&y[i + 4]);
        y_v_2 = vld1q_f32(&y[i + 8]);
        y_v_3 = vld1q_f32(&y[i + 12]);
        x_v_0 = vld1q_f32(&x[i]);
        x_v_1 = vld1q_f32(&x[i + 4]);
        x_v_2 = vld1q_f32(&x[i + 8]);
        x_v_3 = vld1q_f32(&x[i + 12]);
        y_v_0 = vfmaq_f32(y_v_0, x_v_0, val_v);
        y_v_1 = vfmaq_f32(y_v_1, x_v_1, val_v);
        y_v_2 = vfmaq_f32(y_v_2, x_v_2, val_v);
        y_v_3 = vfmaq_f32(y_v_3, x_v_3, val_v);
        vst1q_f32(&y[i], y_v_0);
        vst1q_f32(&y[i + 4], y_v_1);
        vst1q_f32(&y[i + 8], y_v_2);
        vst1q_f32(&y[i + 12], y_v_3);
    }
#endif
    for (; i < len; i += 1)
    {
        y[i] +=  val* x[i];
    }
}
#ifdef __aarch64__
static inline void vec_fma2_tr_s(float *y, const float *x, const float *val, const float diag, const float alpha, ALPHA_INT * col_indx, ALPHA_INT nnzr, ALPHA_INT out_y_col, ALPHA_INT r, ALPHA_INT ldy)
{     
    ALPHA_INT ns4 = ((nnzr >> 2) << 2);
    ALPHA_INT i;

    float32x4_t t0, t1, t2, t3;
    float32x4_t y0, y1, y2, y3;
    float32x4_t v0, v1, v2, v3;
    t0 = vdupq_n_f32(0.0f);
    t1 = vdupq_n_f32(0.0f);
    t2 = vdupq_n_f32(0.0f);
    t3 = vdupq_n_f32(0.0f);

    for (i = 0; i < ns4; i += 4)
    {
        y0 = vld1q_f32(&y[col_indx[i] * ldy + out_y_col]); //ferch 4 values in y
        y1 = vld1q_f32(&y[col_indx[i + 1] * ldy + out_y_col]);
        y2 = vld1q_f32(&y[col_indx[i + 2] * ldy + out_y_col]);
        y3 = vld1q_f32(&y[col_indx[i + 3] * ldy + out_y_col]);
        v0 = vdupq_n_f32(val[i]);
        v1 = vdupq_n_f32(val[i + 1]);
        v2 = vdupq_n_f32(val[i + 2]);
        v3 = vdupq_n_f32(val[i + 3]);

        t0 = vfmaq_f32(t0, y0, v0);
        t1 = vfmaq_f32(t1, y1, v1);
        t2 = vfmaq_f32(t2, y2, v2);
        t3 = vfmaq_f32(t3, y3, v3);
    }

    for (; i < nnzr; ++i)
    {
        y0 = vld1q_f32(&y[col_indx[i] * ldy + out_y_col]);
        v0 = vdupq_n_f32(val[i]);
        t0 = vfmaq_f32(t0, y0, v0);
    }

    t0 = vaddq_f32(t0, t1);
    t2 = vaddq_f32(t2, t3);
    t0 = vaddq_f32(t0, t2);

    float32x4_t x0;
    x0 = vld1q_f32(x); 
    x0 = vmulq_n_f32(x0, alpha);
    x0 = vsubq_f32(x0, t0);

    v0 = vdupq_n_f32(diag);   
    y0 = vdivq_f32(x0, v0);
    vst1q_f32(&y[r * ldy + out_y_col], y0);
}
#endif