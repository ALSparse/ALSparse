#pragma once

#include "../types.h"
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#include "../compute.h"
#define vec_fma2 vec_fma2_z
#define vec_fmai vec_fmai_z
#define vec_fmai_conj vec_fmai_conj_z
#define VEC_FMA2 VEC_FMA2_Z
#define VEC_FMS2 VEC_FMS2_Z
#define VEC_FMS2_CONJ2 VEC_FMS2_Z_CONJ2
#define VEC_MUL2 VEC_MUL2_Z
#define VEC_MUL2_4 VEC_MUL2_Z4

#ifdef __aarch64__
#define VEC_MUL2_Z4(y, x, __val)                                    \
    {                                                               \
        float64x2_t v_val_r = vdupq_n_f64((__val).real);            \
        float64x2_t v_val_i = vdupq_n_f64((__val).imag);            \
        float64x2x2_t v_x, v_x_2, z_v_0, z_v_2;                     \
        float64x2_t xr_valr, xr_vali, xr_valr_2, xr_vali_2;         \
        v_x = vld2q_f64((double *)((x)));                           \
        v_x_2 = vld2q_f64((double *)((x) + 2));                     \
        xr_valr = vmulq_f64(v_x.val[0], v_val_r);                   \
        xr_vali = vmulq_f64(v_x.val[0], v_val_i);                   \
        xr_valr_2 = vmulq_f64(v_x_2.val[0], v_val_r);               \
        xr_vali_2 = vmulq_f64(v_x_2.val[0], v_val_i);               \
        z_v_0.val[0] = vfmsq_f64(xr_valr, v_x.val[1], v_val_i);     \
        z_v_0.val[1] = vfmaq_f64(xr_vali, v_x.val[1], v_val_r);     \
        z_v_2.val[0] = vfmsq_f64(xr_valr_2, v_x_2.val[1], v_val_i); \
        z_v_2.val[1] = vfmaq_f64(xr_vali_2, v_x_2.val[1], v_val_r); \
        vst2q_f64((double *)((y)), z_v_0);                          \
        vst2q_f64((double *)((y) + 2), z_v_2);                      \
    }
#else
#define VEC_MUL2_Z4(y, x, __val) VEC_MUL2_Z(y, x, __val, 4)
#endif

/* y[] = x[] * val */
#ifdef CMLA
#define VEC_MUL2_Z(y, x, __val, len)                                            \
    do                                                                          \
    {                                                                           \
        float64x2_t v_x0, v_x1, v_x2, v_x3;                                     \
        float64x2_t v_y0, v_y1, v_y2, v_y3;                                     \
        float64x2_t v_val = vld1q_f64((double *)&__val);                        \
        float64x2_t zero = vmovq_n_f64(0);                                      \
        ALPHA_INT _i_ = 0;                                                        \
        for (_i_ = 0; _i_ < (len)-3; _i_ += 4)                                  \
        {                                                                       \
            v_x0 = vld1q_f64((double *)(x + _i_));                              \
            v_x1 = vld1q_f64((double *)(x + _i_ + 1));                          \
            v_x2 = vld1q_f64((double *)(x + _i_ + 2));                          \
            v_x3 = vld1q_f64((double *)(x + _i_ + 3));                          \
            (v_y0) = vcmlaq_f64((zero), (v_x0), (v_val));                       \
            (v_y1) = vcmlaq_f64((zero), (v_x1), (v_val));                       \
            (v_y2) = vcmlaq_f64((zero), (v_x2), (v_val));                       \
            (v_y3) = vcmlaq_f64((zero), (v_x3), (v_val));                       \
            (v_y0) = vcmlaq_rot90_f64((v_y0), (v_x0), (v_val));                 \
            (v_y1) = vcmlaq_rot90_f64((v_y1), (v_x1), (v_val));                 \
            (v_y2) = vcmlaq_rot90_f64((v_y2), (v_x2), (v_val));                 \
            (v_y3) = vcmlaq_rot90_f64((v_y3), (v_x3), (v_val));                 \
            vst1q_f64((double *)(y + _i_), v_y0);                               \
            vst1q_f64((double *)(y + _i_ + 1), v_y1);                           \
            vst1q_f64((double *)(y + _i_ + 2), v_y2);                           \
            vst1q_f64((double *)(y + _i_ + 3), v_y3);                           \
        }                                                                       \
        for (; _i_ < len; _i_++)                                                \
        {                                                                       \
            const ALPHA_Complex16 *__X = (x) + _i_;                               \
            ALPHA_Complex16 *__Y = (y) + _i_;                                     \
            ALPHA_Float real = (__X)->real * __val.real - __X->imag * __val.imag; \
            __Y->imag = (__X)->imag * __val.real + __X->real * __val.imag;      \
            __Y->real = real;                                                   \
        }                                                                       \
    } while (0)
#else
#define VEC_MUL2_Z(y, x, __val, len)                                                                   \
    do                                                                                                 \
    {                                                                                                  \
        int32_t __i = 0;                                                                               \
        for (; __i < len; __i += 1)                                                                    \
        {                                                                                              \
            double _B = (x)[__i].real * (__val).imag;                                                  \
            double _Z = (x)[__i].imag * (__val).real;                                                  \
            (y)[__i].real = ((x)[__i].real + (x)[__i].imag) * ((__val).real - (__val).imag) + _B - _Z; \
            (y)[__i].imag = _B + _Z;                                                                   \
        }                                                                                              \
    } while (0)
#endif

#ifdef CMLA
#define VEC_FMS2_Z(y, x, __val, len)                                                        \
    do                                                                                      \
    {                                                                                       \
        float64x2_t v_x0, v_x1, v_x2, v_x3;                                                 \
        float64x2_t v_y0, v_y1, v_y2, v_y3;                                                 \
        float64x2_t v_val = vld1q_f64((double *)&__val);                                    \
        ALPHA_INT _i_ = 0;                                                                    \
        for (_i_ = 0; _i_ < (len)-3; _i_ += 4)                                              \
        {                                                                                   \
            v_x0 = vld1q_f64((double *)(x + _i_));                                          \
            v_x1 = vld1q_f64((double *)(x + _i_ + 1));                                      \
            v_x2 = vld1q_f64((double *)(x + _i_ + 2));                                      \
            v_x3 = vld1q_f64((double *)(x + _i_ + 3));                                      \
            v_y0 = vld1q_f64((double *)(y + _i_));                                          \
            v_y1 = vld1q_f64((double *)(y + _i_ + 1));                                      \
            v_y2 = vld1q_f64((double *)(y + _i_ + 2));                                      \
            v_y3 = vld1q_f64((double *)(y + _i_ + 3));                                      \
            (v_y0) = vcmlaq_rot180_f64((v_y0), (v_x0), (v_val));                            \
            (v_y1) = vcmlaq_rot180_f64((v_y1), (v_x1), (v_val));                            \
            (v_y2) = vcmlaq_rot180_f64((v_y2), (v_x2), (v_val));                            \
            (v_y3) = vcmlaq_rot180_f64((v_y3), (v_x3), (v_val));                            \
            (v_y0) = vcmlaq_rot270_f64((v_y0), (v_x0), (v_val));                            \
            (v_y1) = vcmlaq_rot270_f64((v_y1), (v_x1), (v_val));                            \
            (v_y2) = vcmlaq_rot270_f64((v_y2), (v_x2), (v_val));                            \
            (v_y3) = vcmlaq_rot270_f64((v_y3), (v_x3), (v_val));                            \
            vst1q_f64((double *)(y + _i_), v_y0);                                           \
            vst1q_f64((double *)(y + _i_ + 1), v_y1);                                       \
            vst1q_f64((double *)(y + _i_ + 2), v_y2);                                       \
            vst1q_f64((double *)(y + _i_ + 3), v_y3);                                       \
        }                                                                                   \
        for (; _i_ < len; ++_i_)                                                            \
        {                                                                                   \
            const ALPHA_Complex16 *__X = (x) + _i_;                                           \
            ALPHA_Complex16 *__Y = (y) + _i_;                                                 \
            ALPHA_Float real = __Y->real - (__X)->real * __val.real + __X->imag * __val.imag; \
            __Y->imag -= (__X)->imag * __val.real + __X->real * __val.imag;                 \
            __Y->real = real;                                                               \
        }                                                                                   \
    } while (0)
#else
#define VEC_FMS2_Z(y, x, __val, len)                                                                    \
    do                                                                                                  \
    {                                                                                                   \
        int32_t __i = 0;                                                                                \
        for (; __i < len; __i += 1)                                                                     \
        {                                                                                               \
            double _B = (x)[__i].real * (__val).imag;                                                   \
            double _Z = (x)[__i].imag * (__val).real;                                                   \
            (y)[__i].real -= ((x)[__i].real + (x)[__i].imag) * ((__val).real - (__val).imag) + _B - _Z; \
            (y)[__i].imag -= _B + _Z;                                                                   \
        }                                                                                               \
    } while (0)
#endif

#ifdef CMLA
#define VEC_FMS2_Z_CONJ2(y, x, __val, len)                                                        \
    do                                                                                      \
    {                                                                                       \
        float64x2_t v_x0, v_x1, v_x2, v_x3;                                                 \
        float64x2_t v_y0, v_y1, v_y2, v_y3;                                                 \
        float64x2_t v_val = vld1q_f64((double *)&__val);                                    \
        ALPHA_INT _i_ = 0;                                                                    \
        for (_i_ = 0; _i_ < (len)-3; _i_ += 4)                                              \
        {                                                                                   \
            v_x0 = vld1q_f64((double *)(x + _i_));                                          \
            v_x1 = vld1q_f64((double *)(x + _i_ + 1));                                      \
            v_x2 = vld1q_f64((double *)(x + _i_ + 2));                                      \
            v_x3 = vld1q_f64((double *)(x + _i_ + 3));                                      \
            v_y0 = vld1q_f64((double *)(y + _i_));                                          \
            v_y1 = vld1q_f64((double *)(y + _i_ + 1));                                      \
            v_y2 = vld1q_f64((double *)(y + _i_ + 2));                                      \
            v_y3 = vld1q_f64((double *)(y + _i_ + 3));                                      \
            (v_y0) = vcmlaq_rot180_f64((v_y0), (v_x0), (v_val));                            \
            (v_y1) = vcmlaq_rot180_f64((v_y1), (v_x1), (v_val));                            \
            (v_y2) = vcmlaq_rot180_f64((v_y2), (v_x2), (v_val));                            \
            (v_y3) = vcmlaq_rot180_f64((v_y3), (v_x3), (v_val));                            \
            (v_y0) = vcmlaq_rot90_f64((v_y0), (v_x0), (v_val));                            \
            (v_y1) = vcmlaq_rot90_f64((v_y1), (v_x1), (v_val));                            \
            (v_y2) = vcmlaq_rot90_f64((v_y2), (v_x2), (v_val));                            \
            (v_y3) = vcmlaq_rot90_f64((v_y3), (v_x3), (v_val));                            \
            vst1q_f64((double *)(y + _i_), v_y0);                                           \
            vst1q_f64((double *)(y + _i_ + 1), v_y1);                                       \
            vst1q_f64((double *)(y + _i_ + 2), v_y2);                                       \
            vst1q_f64((double *)(y + _i_ + 3), v_y3);                                       \
        }                                                                                   \
        for (; _i_ < len; ++_i_)                                                            \
        {                                                                                   \
            const ALPHA_Complex16 *__X = (x) + _i_;                                           \
            ALPHA_Complex16 *__Y = (y) + _i_;                                                 \
            ALPHA_Float real = __Y->real - (__X)->real * __val.real - __X->imag * __val.imag; \
            __Y->imag -= -(__X)->imag * __val.real + __X->real * __val.imag;                 \
            __Y->real = real;                                                               \
        }                                                                                   \
    } while (0)
#else
#define VEC_FMS2_Z_CONJ2(y, x, __val, len)                                                                    \
    do                                                                                                  \
    {                                                                                                   \
        int32_t __i = 0;                                                                                \
        for (; __i < len; __i += 1)                                                                     \
        {                                                                                               \
            double _B = (x)[__i].real * (__val).imag;                                                   \
            double _Z = -(x)[__i].imag * (__val).real;                                                   \
            (y)[__i].real -= ((x)[__i].real - (x)[__i].imag) * ((__val).real - (__val).imag) + _B - _Z; \
            (y)[__i].imag -= _B + _Z;                                                                   \
        }                                                                                               \
    } while (0)
#endif

// y[] += x[] * val
#ifdef CMLA
#define VEC_FMA2_Z(y, x, __val, len)                                                                                    \
    do                                                                                                                  \
    {                                                                                                                   \
        float64x2_t v_tmp0 = vdupq_n_f64(0), v_tmp1 = vdupq_n_f64(0), v_tmp2 = vdupq_n_f64(0), v_tmp3 = vdupq_n_f64(0); \
        float64x2_t v_x0, v_x1, v_x2, v_x3;                                                                             \
        float64x2_t v_y0, v_y1, v_y2, v_y3;                                                                             \
        float64x2_t v_val = vld1q_f64((double *)&__val);                                                                \
        ALPHA_INT __i = 0;                                                                                                \
        for (__i = 0; __i < (len)-3; __i += 4)                                                                          \
        {                                                                                                               \
            v_x0 = vld1q_f64((double *)(x + __i));                                                                      \
            v_x1 = vld1q_f64((double *)(x + __i + 1));                                                                  \
            v_x2 = vld1q_f64((double *)(x + __i + 2));                                                                  \
            v_x3 = vld1q_f64((double *)(x + __i + 3));                                                                  \
            v_y0 = vld1q_f64((double *)(y + __i));                                                                      \
            v_y1 = vld1q_f64((double *)(y + __i + 1));                                                                  \
            v_y2 = vld1q_f64((double *)(y + __i + 2));                                                                  \
            v_y3 = vld1q_f64((double *)(y + __i + 3));                                                                  \
            (v_y0) = vcmlaq_f64((v_y0), (v_x0), (v_val));                                                               \
            (v_y1) = vcmlaq_f64((v_y1), (v_x1), (v_val));                                                               \
            (v_y2) = vcmlaq_f64((v_y2), (v_x2), (v_val));                                                               \
            (v_y3) = vcmlaq_f64((v_y3), (v_x3), (v_val));                                                               \
            (v_y0) = vcmlaq_rot90_f64((v_y0), (v_x0), (v_val));                                                         \
            (v_y1) = vcmlaq_rot90_f64((v_y1), (v_x1), (v_val));                                                         \
            (v_y2) = vcmlaq_rot90_f64((v_y2), (v_x2), (v_val));                                                         \
            (v_y3) = vcmlaq_rot90_f64((v_y3), (v_x3), (v_val));                                                         \
            vst1q_f64((double *)(y + __i), v_y0);                                                                       \
            vst1q_f64((double *)(y + __i + 1), v_y1);                                                                   \
            vst1q_f64((double *)(y + __i + 2), v_y2);                                                                   \
            vst1q_f64((double *)(y + __i + 3), v_y3);                                                                   \
        }                                                                                                               \
        for (; __i < (len); __i++)                                                                                      \
        {                                                                                                               \
            const ALPHA_Complex16 *__X = (x) + __i;                                                                       \
            ALPHA_Complex16 *__Y = (y) + __i;                                                                             \
            ALPHA_Float real = __Y->real + (__X)->real * __val.real - __X->imag * __val.imag;                             \
            __Y->imag += (__X)->imag * __val.real + __X->real * __val.imag;                                             \
            __Y->real = real;                                                                                           \
        }                                                                                                               \
    } while (0)
#else
#define VEC_FMA2_Z(y, x, __val, len)                                                                    \
    do                                                                                                  \
    {                                                                                                   \
        int32_t __i = 0;                                                                                \
        for (; __i < len; __i += 1)                                                                     \
        {                                                                                               \
            double _B = (x)[__i].real * (__val).imag;                                                   \
            double _Z = (x)[__i].imag * (__val).real;                                                   \
            (y)[__i].real += ((x)[__i].real + (x)[__i].imag) * ((__val).real - (__val).imag) + _B - _Z; \
            (y)[__i].imag += _B + _Z;                                                                   \
        }                                                                                               \
    } while (0)
#endif

static inline void vec_fmai_z(ALPHA_Complex16 *y, const ALPHA_INT *indx, const ALPHA_Complex16 *x, const ALPHA_INT ns, const ALPHA_Complex16 val)
{
    ALPHA_INT ns4 = ((ns >> 2) << 2);
    ALPHA_INT i = 0;
#if defined(CMLA)
    float64x2_t v_tmp0 = vdupq_n_f64(0), v_tmp1 = vdupq_n_f64(0), v_tmp2 = vdupq_n_f64(0), v_tmp3 = vdupq_n_f64(0);
    float64x2_t v_x0, v_x1, v_x2, v_x3;
    float64x2_t v_y0, v_y1, v_y2, v_y3;
    float64x2_t v_val = vld1q_f64((double *)&val);
#endif
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
#if defined(CMLA)
        v_x0 = vld1q_f64((double *)(x + i));
        v_x1 = vld1q_f64((double *)(x + i + 1));
        v_x2 = vld1q_f64((double *)(x + i + 2));
        v_x3 = vld1q_f64((double *)(x + i + 3));

        v_y0 = vld1q_f64((double *)(y + indx[i]));
        v_y1 = vld1q_f64((double *)(y + indx[i + 1]));
        v_y2 = vld1q_f64((double *)(y + indx[i + 2]));
        v_y3 = vld1q_f64((double *)(y + indx[i + 3]));

        (v_y0) = vcmlaq_f64((v_y0), (v_x0), (v_val));
        (v_y1) = vcmlaq_f64((v_y1), (v_x1), (v_val));
        (v_y2) = vcmlaq_f64((v_y2), (v_x2), (v_val));
        (v_y3) = vcmlaq_f64((v_y3), (v_x3), (v_val));

        (v_y0) = vcmlaq_rot90_f64((v_y0), (v_x0), (v_val));
        (v_y1) = vcmlaq_rot90_f64((v_y1), (v_x1), (v_val));
        (v_y2) = vcmlaq_rot90_f64((v_y2), (v_x2), (v_val));
        (v_y3) = vcmlaq_rot90_f64((v_y3), (v_x3), (v_val));

        vst1q_f64((double *)(y + indx[i]), v_y0);
        vst1q_f64((double *)(y + indx[i + 1]), v_y1);
        vst1q_f64((double *)(y + indx[i + 2]), v_y2);
        vst1q_f64((double *)(y + indx[i + 3]), v_y3);
#else
        cmp_madde(y[indx[i]], x[i], val);
        cmp_madde(y[indx[i + 1]], x[i + 1], val);
        cmp_madde(y[indx[i + 2]], x[i + 2], val);
        cmp_madde(y[indx[i + 3]], x[i + 3], val);
#endif
    }
    for (; i < ns; ++i)
    {
        cmp_madde(y[indx[i]], x[i], val);
    }
}
static inline void vec_fmai_conj_z(ALPHA_Complex16 *y, const ALPHA_INT *indx, const ALPHA_Complex16 *x, const ALPHA_INT ns, const ALPHA_Complex16 val)
{
    ALPHA_INT ns4 = ((ns >> 2) << 2);
    ALPHA_INT i = 0;
#if defined(CMLA)
    float64x2_t v_x0, v_x1, v_x2, v_x3;
    float64x2_t v_y0, v_y1, v_y2, v_y3;
    float64x2_t v_val = vld1q_f64((double *)&val);
#endif
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
#if defined(CMLA)
        v_x0 = vld1q_f64((double *)(x + i));
        v_x1 = vld1q_f64((double *)(x + i + 1));
        v_x2 = vld1q_f64((double *)(x + i + 2));
        v_x3 = vld1q_f64((double *)(x + i + 3));

        v_y0 = vld1q_f64((double *)(y + indx[i]));
        v_y1 = vld1q_f64((double *)(y + indx[i + 1]));
        v_y2 = vld1q_f64((double *)(y + indx[i + 2]));
        v_y3 = vld1q_f64((double *)(y + indx[i + 3]));

        (v_y0) = vcmlaq_f64((v_y0), (v_x0), (v_val));
        (v_y1) = vcmlaq_f64((v_y1), (v_x1), (v_val));
        (v_y2) = vcmlaq_f64((v_y2), (v_x2), (v_val));
        (v_y3) = vcmlaq_f64((v_y3), (v_x3), (v_val));

        (v_y0) = vcmlaq_rot270_f64((v_y0), (v_x0), (v_val));
        (v_y1) = vcmlaq_rot270_f64((v_y1), (v_x1), (v_val));
        (v_y2) = vcmlaq_rot270_f64((v_y2), (v_x2), (v_val));
        (v_y3) = vcmlaq_rot270_f64((v_y3), (v_x3), (v_val));

        vst1q_f64((double *)(y + indx[i]), v_y0);
        vst1q_f64((double *)(y + indx[i + 1]), v_y1);
        vst1q_f64((double *)(y + indx[i + 2]), v_y2);
        vst1q_f64((double *)(y + indx[i + 3]), v_y3);
#else
        cmp_madde_2c(y[indx[i]], x[i], val);
        cmp_madde_2c(y[indx[i + 1]], x[i + 1], val);
        cmp_madde_2c(y[indx[i + 2]], x[i + 2], val);
        cmp_madde_2c(y[indx[i + 3]], x[i + 3], val);
#endif
    }
    for (; i < ns; ++i)
    {
        cmp_madde_2c(y[indx[i]], x[i], val);
    }
}
static inline void vec_fma2_z(ALPHA_Complex16 *y, const ALPHA_Complex16 *x, ALPHA_Complex16 val, ALPHA_INT len)
{
    ALPHA_INT ns4 = ((len >> 2) << 2);
    ALPHA_INT i = 0;
#if defined(CMLA)
    float64x2_t v_tmp0 = vdupq_n_f64(0), v_tmp1 = vdupq_n_f64(0), v_tmp2 = vdupq_n_f64(0), v_tmp3 = vdupq_n_f64(0);
    float64x2_t v_x0, v_x1, v_x2, v_x3;
    float64x2_t v_y0, v_y1, v_y2, v_y3;
    float64x2_t v_val = vld1q_f64((double *)&val);
#endif

    for (i = 0; i < ns4; i += 4)
    {
#if defined(CMLA)
        v_x0 = vld1q_f64((double *)(x + i));
        v_x1 = vld1q_f64((double *)(x + i + 1));
        v_x2 = vld1q_f64((double *)(x + i + 2));
        v_x3 = vld1q_f64((double *)(x + i + 3));

        v_y0 = vld1q_f64((double *)(y + i));
        v_y1 = vld1q_f64((double *)(y + i + 1));
        v_y2 = vld1q_f64((double *)(y + i + 2));
        v_y3 = vld1q_f64((double *)(y + i + 3));

        (v_y0) = vcmlaq_f64((v_y0), (v_x0), (v_val));
        (v_y1) = vcmlaq_f64((v_y1), (v_x1), (v_val));
        (v_y2) = vcmlaq_f64((v_y2), (v_x2), (v_val));
        (v_y3) = vcmlaq_f64((v_y3), (v_x3), (v_val));

        (v_y0) = vcmlaq_rot90_f64((v_y0), (v_x0), (v_val));
        (v_y1) = vcmlaq_rot90_f64((v_y1), (v_x1), (v_val));
        (v_y2) = vcmlaq_rot90_f64((v_y2), (v_x2), (v_val));
        (v_y3) = vcmlaq_rot90_f64((v_y3), (v_x3), (v_val));

        vst1q_f64((double *)(y + i), v_y0);
        vst1q_f64((double *)(y + i + 1), v_y1);
        vst1q_f64((double *)(y + i + 2), v_y2);
        vst1q_f64((double *)(y + i + 3), v_y3);
#else
        cmp_madde(y[i], x[i], val);
        cmp_madde(y[i + 1], x[i + 1], val);
        cmp_madde(y[i + 2], x[i + 2], val);
        cmp_madde(y[i + 3], x[i + 3], val);
#endif
    }
    for (; i < len; ++i)
    {
        cmp_madde(y[i], x[i], val);
    }
}