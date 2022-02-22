#pragma once

#include "../types.h"
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#include "../compute.h"
#define vec_fma2 vec_fma2_c
#define vec_fmai vec_fmai_c
#define vec_fmai_conj vec_fmai_conj_c
#define VEC_FMA2 VEC_FMA2_C
#define VEC_FMS2 VEC_FMS2_C
#define VEC_FMS2_CONJ2 VEC_FMS2_C_CONJ2
#define VEC_MUL2 VEC_MUL2_C
#define VEC_MUL2_4 VEC_MUL2_C4

#ifdef __aarch64__
#define VEC_MUL2_C4(y, x, __val)                                  \
    {                                                             \
        float32x4_t v_val_r = vdupq_n_f32((__val).real);          \
        float32x4_t v_val_i = vdupq_n_f32((__val).imag);          \
        float32x4x2_t z_v_0;                                      \
        float32x4x2_t v_x0 = vld2q_f32((float *)((x)));           \
        float32x4_t xr_valr0 = vmulq_f32(v_x0.val[0], v_val_r);   \
        float32x4_t xr_vali0 = vmulq_f32(v_x0.val[0], v_val_i);   \
        z_v_0.val[0] = vfmsq_f32(xr_valr0, v_x0.val[1], v_val_i); \
        z_v_0.val[1] = vfmaq_f32(xr_vali0, v_x0.val[1], v_val_r); \
        vst2q_f32((float *)((y)), z_v_0);                         \
    }
#else
#define VEC_MUL2_C4(y, x, __val) VEC_MUL2_C(y, x, __val, 4)
#endif

// y[] = x[] * val
#ifdef CMLA
#define VEC_MUL2_C(y, x, __val, len)                                            \
    do                                                                          \
    {                                                                           \
        float32x4_t v_x0, v_x1, v_x2, v_x3;                                     \
        float32x4_t v_y0, v_y1, v_y2, v_y3;                                     \
        float64x2_t v_val0 = vld1q_dup_f64((void *)&__val);                     \
        float32x4_t v_val = vreinterpretq_f32_f64(v_val0);                      \
        float32x4_t zero = vmovq_n_f32(0);                                      \
        ALPHA_INT _i_ = 0;                                                        \
        for (_i_ = 0; _i_ < (len)-7; _i_ += 8)                                  \
        {                                                                       \
            v_x0 = vld1q_f32((float *)(x + _i_));                               \
            v_x1 = vld1q_f32((float *)(x + _i_ + 2));                           \
            v_x2 = vld1q_f32((float *)(x + _i_ + 4));                           \
            v_x3 = vld1q_f32((float *)(x + _i_ + 6));                           \
            (v_y0) = vcmlaq_f32((zero), (v_x0), (v_val));                       \
            (v_y1) = vcmlaq_f32((zero), (v_x1), (v_val));                       \
            (v_y2) = vcmlaq_f32((zero), (v_x2), (v_val));                       \
            (v_y3) = vcmlaq_f32((zero), (v_x3), (v_val));                       \
            (v_y0) = vcmlaq_rot90_f32((v_y0), (v_x0), (v_val));                 \
            (v_y1) = vcmlaq_rot90_f32((v_y1), (v_x1), (v_val));                 \
            (v_y2) = vcmlaq_rot90_f32((v_y2), (v_x2), (v_val));                 \
            (v_y3) = vcmlaq_rot90_f32((v_y3), (v_x3), (v_val));                 \
            vst1q_f32((float *)(y + _i_), v_y0);                                \
            vst1q_f32((float *)(y + _i_ + 2), v_y1);                            \
            vst1q_f32((float *)(y + _i_ + 4), v_y2);                            \
            vst1q_f32((float *)(y + _i_ + 6), v_y3);                            \
        }                                                                       \
        for (; _i_ < len; ++_i_)                                                \
        {                                                                       \
            const ALPHA_Complex8 *__X = (x) + _i_;                                \
            ALPHA_Complex8 *__Y = (y) + _i_;                                      \
            ALPHA_Float real = (__X)->real * __val.real - __X->imag * __val.imag; \
            __Y->imag = (__X)->imag * __val.real + __X->real * __val.imag;      \
            __Y->real = real;                                                   \
        }                                                                       \
    } while (0)
#else
#define VEC_MUL2_C(y, x, __val, len)                                                                   \
    do                                                                                                 \
    {                                                                                                  \
        int32_t __i = 0;                                                                               \
        for (; __i < len; __i += 1)                                                                    \
        {                                                                                              \
            float _B = (x)[__i].real * (__val).imag;                                                   \
            float _C = (x)[__i].imag * (__val).real;                                                   \
            (y)[__i].real = ((x)[__i].real + (x)[__i].imag) * ((__val).real - (__val).imag) + _B - _C; \
            (y)[__i].imag = _B + _C;                                                                   \
        }                                                                                              \
    } while (0)
#endif

// y[] -= x[] * val
#ifdef CMLA
#define VEC_FMS2_C(y, x, __val, len)                                                                                    \
    do                                                                                                                  \
    {                                                                                                                   \
        float32x4_t v_tmp0 = vdupq_n_f32(0), v_tmp1 = vdupq_n_f32(0), v_tmp2 = vdupq_n_f32(0), v_tmp3 = vdupq_n_f32(0); \
        float32x4_t v_x0, v_x1, v_x2, v_x3;                                                                             \
        float32x4_t v_y0, v_y1, v_y2, v_y3;                                                                             \
        float64x2_t v_val0 = vld1q_dup_f64((void *)&__val);                                                             \
        float32x4_t v_val = vreinterpretq_f32_f64(v_val0);                                                              \
        ALPHA_INT _i_ = 0;                                                                                                \
        for (_i_ = 0; _i_ < (len)-7; _i_ += 8)                                                                          \
        {                                                                                                               \
            v_x0 = vld1q_f32((float *)(x + _i_));                                                                       \
            v_x1 = vld1q_f32((float *)(x + _i_ + 2));                                                                   \
            v_x2 = vld1q_f32((float *)(x + _i_ + 4));                                                                   \
            v_x3 = vld1q_f32((float *)(x + _i_ + 6));                                                                   \
            v_y0 = vld1q_f32((float *)(y + _i_));                                                                       \
            v_y1 = vld1q_f32((float *)(y + _i_ + 2));                                                                   \
            v_y2 = vld1q_f32((float *)(y + _i_ + 4));                                                                   \
            v_y3 = vld1q_f32((float *)(y + _i_ + 6));                                                                   \
            (v_y0) = vcmlaq_rot180_f32((v_y0), (v_x0), (v_val));                                                        \
            (v_y1) = vcmlaq_rot180_f32((v_y1), (v_x1), (v_val));                                                        \
            (v_y2) = vcmlaq_rot180_f32((v_y2), (v_x2), (v_val));                                                        \
            (v_y3) = vcmlaq_rot180_f32((v_y3), (v_x3), (v_val));                                                        \
            (v_y0) = vcmlaq_rot270_f32((v_y0), (v_x0), (v_val));                                                        \
            (v_y1) = vcmlaq_rot270_f32((v_y1), (v_x1), (v_val));                                                        \
            (v_y2) = vcmlaq_rot270_f32((v_y2), (v_x2), (v_val));                                                        \
            (v_y3) = vcmlaq_rot270_f32((v_y3), (v_x3), (v_val));                                                        \
            vst1q_f32((float *)(y + _i_), v_y0);                                                                        \
            vst1q_f32((float *)(y + _i_ + 2), v_y1);                                                                    \
            vst1q_f32((float *)(y + _i_ + 4), v_y2);                                                                    \
            vst1q_f32((float *)(y + _i_ + 6), v_y3);                                                                    \
        }                                                                                                               \
        for (; _i_ < (len)-3; _i_ += 4)                                                                                 \
        {                                                                                                               \
            v_x0 = vld1q_f32((float *)(x + _i_));                                                                       \
            v_x1 = vld1q_f32((float *)(x + _i_ + 2));                                                                   \
            v_y0 = vld1q_f32((float *)(y + _i_));                                                                       \
            v_y1 = vld1q_f32((float *)(y + _i_ + 2));                                                                   \
            (v_y0) = vcmlaq_rot180_f32((v_y0), (v_x0), (v_val));                                                        \
            (v_y1) = vcmlaq_rot180_f32((v_y1), (v_x1), (v_val));                                                        \
            (v_y0) = vcmlaq_rot270_f32((v_y0), (v_x0), (v_val));                                                        \
            (v_y1) = vcmlaq_rot270_f32((v_y1), (v_x1), (v_val));                                                        \
            vst1q_f32((float *)(y + _i_), v_y0);                                                                        \
            vst1q_f32((float *)(y + _i_ + 2), v_y1);                                                                    \
        }                                                                                                               \
        for (; _i_ < len; ++_i_)                                                                                        \
        {                                                                                                               \
            const ALPHA_Complex8 *__X = (x) + _i_;                                                                        \
            ALPHA_Complex8 *__Y = (y) + _i_;                                                                              \
            ALPHA_Float real = __Y->real - (__X)->real * __val.real + __X->imag * __val.imag;                             \
            __Y->imag -= (__X)->imag * __val.real + __X->real * __val.imag;                                             \
            __Y->real = real;                                                                                           \
        }                                                                                                               \
    } while (0)
#else
#define VEC_FMS2_C(y, x, __val, len)                                                                    \
    do                                                                                                  \
    {                                                                                                   \
        int32_t __i = 0;                                                                                \
        for (; __i < len; __i += 1)                                                                     \
        {                                                                                               \
            float _B = (x)[__i].real * (__val).imag;                                                    \
            float _C = (x)[__i].imag * (__val).real;                                                    \
            (y)[__i].real -= ((x)[__i].real + (x)[__i].imag) * ((__val).real - (__val).imag) + _B - _C; \
            (y)[__i].imag -= _B + _C;                                                                   \
        }                                                                                               \
    } while (0)
#endif
// y[] -= conj(x[]) * val
#ifdef CMLA
#define VEC_FMS2_C_CONJ2(y, x, __val, len)                                                                              \
    do                                                                                                                  \
    {                                                                                                                   \
        float32x4_t v_tmp0 = vdupq_n_f32(0), v_tmp1 = vdupq_n_f32(0), v_tmp2 = vdupq_n_f32(0), v_tmp3 = vdupq_n_f32(0); \
        float32x4_t v_x0, v_x1, v_x2, v_x3;                                                                             \
        float32x4_t v_y0, v_y1, v_y2, v_y3;                                                                             \
        float64x2_t v_val0 = vld1q_dup_f64((void *)&__val);                                                             \
        float32x4_t v_val = vreinterpretq_f32_f64(v_val0);                                                              \
        ALPHA_INT _i_ = 0;                                                                                                \
        for (_i_ = 0; _i_ < (len)-7; _i_ += 8)                                                                          \
        {                                                                                                               \
            v_x0 = vld1q_f32((float *)(x + _i_));                                                                       \
            v_x1 = vld1q_f32((float *)(x + _i_ + 2));                                                                   \
            v_x2 = vld1q_f32((float *)(x + _i_ + 4));                                                                   \
            v_x3 = vld1q_f32((float *)(x + _i_ + 6));                                                                   \
            v_y0 = vld1q_f32((float *)(y + _i_));                                                                       \
            v_y1 = vld1q_f32((float *)(y + _i_ + 2));                                                                   \
            v_y2 = vld1q_f32((float *)(y + _i_ + 4));                                                                   \
            v_y3 = vld1q_f32((float *)(y + _i_ + 6));                                                                   \
            (v_y0) = vcmlaq_rot180_f32((v_y0), (v_x0), (v_val));                                                        \
            (v_y1) = vcmlaq_rot180_f32((v_y1), (v_x1), (v_val));                                                        \
            (v_y2) = vcmlaq_rot180_f32((v_y2), (v_x2), (v_val));                                                        \
            (v_y3) = vcmlaq_rot180_f32((v_y3), (v_x3), (v_val));                                                        \
            (v_y0) = vcmlaq_rot90_f32((v_y0), (v_x0), (v_val));                                                         \
            (v_y1) = vcmlaq_rot90_f32((v_y1), (v_x1), (v_val));                                                         \
            (v_y2) = vcmlaq_rot90_f32((v_y2), (v_x2), (v_val));                                                         \
            (v_y3) = vcmlaq_rot90_f32((v_y3), (v_x3), (v_val));                                                         \
            vst1q_f32((float *)(y + _i_), v_y0);                                                                        \
            vst1q_f32((float *)(y + _i_ + 2), v_y1);                                                                    \
            vst1q_f32((float *)(y + _i_ + 4), v_y2);                                                                    \
            vst1q_f32((float *)(y + _i_ + 6), v_y3);                                                                    \
        }                                                                                                               \
        for (; _i_ < (len)-3; _i_ += 4)                                                                                 \
        {                                                                                                               \
            v_x0 = vld1q_f32((float *)(x + _i_));                                                                       \
            v_x1 = vld1q_f32((float *)(x + _i_ + 2));                                                                   \
            v_y0 = vld1q_f32((float *)(y + _i_));                                                                       \
            v_y1 = vld1q_f32((float *)(y + _i_ + 2));                                                                   \
            (v_y0) = vcmlaq_rot180_f32((v_y0), (v_x0), (v_val));                                                        \
            (v_y1) = vcmlaq_rot180_f32((v_y1), (v_x1), (v_val));                                                        \
            (v_y0) = vcmlaq_rot90_f32((v_y0), (v_x0), (v_val));                                                        \
            (v_y1) = vcmlaq_rot90_f32((v_y1), (v_x1), (v_val));                                                        \
            vst1q_f32((float *)(y + _i_), v_y0);                                                                        \
            vst1q_f32((float *)(y + _i_ + 2), v_y1);                                                                    \
        }                                                                                                               \
        for (; _i_ < len; ++_i_)                                                                                        \
        {                                                                                                               \
            const ALPHA_Complex8 *__X = (x) + _i_;                                                                        \
            ALPHA_Complex8 *__Y = (y) + _i_;                                                                              \
            ALPHA_Float real = __Y->real - (__X)->real * __val.real - __X->imag * __val.imag;                             \
            __Y->imag -= -(__X)->imag * __val.real + __X->real * __val.imag;                                             \
            __Y->real = real;                                                                                           \
        }                                                                                                               \
    } while (0)
#else
#define VEC_FMS2_C_CONJ2(y, x, __val, len)                                                                    \
    do                                                                                                  \
    {                                                                                                   \
        int32_t __i = 0;                                                                                \
        for (; __i < len; __i += 1)                                                                     \
        {                                                                                               \
            float _B = (x)[__i].real * (__val).imag;                                                    \
            float _C = -(x)[__i].imag * (__val).real;                                                    \
            (y)[__i].real -= ((x)[__i].real - (x)[__i].imag) * ((__val).real - (__val).imag) + _B - _C; \
            (y)[__i].imag -= _B + _C;                                                                   \
        }                                                                                               \
    } while (0)
#endif

// y[] += x[] * val
#ifdef CMLA
#define VEC_FMA2_C(y, x, __val, len)                                                                                    \
    {                                                                                                                   \
        float32x4_t v_tmp0 = vdupq_n_f32(0), v_tmp1 = vdupq_n_f32(0), v_tmp2 = vdupq_n_f32(0), v_tmp3 = vdupq_n_f32(0); \
        float32x4_t v_x0, v_x1, v_x2, v_x3;                                                                             \
        float32x4_t v_y0, v_y1, v_y2, v_y3;                                                                             \
        float64x2_t v_val0 = vld1q_dup_f64((void *)&__val);                                                             \
        float32x4_t v_val = vreinterpretq_f32_f64(v_val0);                                                              \
        ALPHA_INT _i_ = 0;                                                                                                \
        for (_i_ = 0; _i_ < (len)-7; _i_ += 8)                                                                          \
        {                                                                                                               \
            v_x0 = vld1q_f32((float *)(x + _i_));                                                                       \
            v_x1 = vld1q_f32((float *)(x + _i_ + 2));                                                                   \
            v_x2 = vld1q_f32((float *)(x + _i_ + 4));                                                                   \
            v_x3 = vld1q_f32((float *)(x + _i_ + 6));                                                                   \
            v_y0 = vld1q_f32((float *)(y + _i_));                                                                       \
            v_y1 = vld1q_f32((float *)(y + _i_ + 2));                                                                   \
            v_y2 = vld1q_f32((float *)(y + _i_ + 4));                                                                   \
            v_y3 = vld1q_f32((float *)(y + _i_ + 6));                                                                   \
            (v_y0) = vcmlaq_f32((v_y0), (v_x0), (v_val));                                                               \
            (v_y1) = vcmlaq_f32((v_y1), (v_x1), (v_val));                                                               \
            (v_y2) = vcmlaq_f32((v_y2), (v_x2), (v_val));                                                               \
            (v_y3) = vcmlaq_f32((v_y3), (v_x3), (v_val));                                                               \
            (v_y0) = vcmlaq_rot90_f32((v_y0), (v_x0), (v_val));                                                         \
            (v_y1) = vcmlaq_rot90_f32((v_y1), (v_x1), (v_val));                                                         \
            (v_y2) = vcmlaq_rot90_f32((v_y2), (v_x2), (v_val));                                                         \
            (v_y3) = vcmlaq_rot90_f32((v_y3), (v_x3), (v_val));                                                         \
            vst1q_f32((float *)(y + _i_), v_y0);                                                                        \
            vst1q_f32((float *)(y + _i_ + 2), v_y1);                                                                    \
            vst1q_f32((float *)(y + _i_ + 4), v_y2);                                                                    \
            vst1q_f32((float *)(y + _i_ + 6), v_y3);                                                                    \
        }                                                                                                               \
        for (; _i_ < len; ++_i_)                                                                                        \
        {                                                                                                               \
            const ALPHA_Complex8 *__X = (x) + _i_;                                                                        \
            ALPHA_Complex8 *__Y = (y) + _i_;                                                                              \
            ALPHA_Float real = __Y->real + (__X)->real * __val.real - __X->imag * __val.imag;                             \
            __Y->imag += (__X)->imag * __val.real + __X->real * __val.imag;                                             \
            __Y->real = real;                                                                                           \
        }                                                                                                               \
    }
#else
#define VEC_FMA2_C(y, x, __val, len)                                                                    \
    do                                                                                                  \
    {                                                                                                   \
        int32_t __i = 0;                                                                                \
        for (; __i < len; __i += 1)                                                                     \
        {                                                                                               \
            float _B = (x)[__i].real * (__val).imag;                                                    \
            float _C = (x)[__i].imag * (__val).real;                                                    \
            (y)[__i].real += ((x)[__i].real + (x)[__i].imag) * ((__val).real - (__val).imag) + _B - _C; \
            (y)[__i].imag += _B + _C;                                                                   \
        }                                                                                               \
    } while (0)
#endif

//scatter: y[indx[:]] += x[:] * val, noting x is continuous
static inline void vec_fmai_c(ALPHA_Complex8 *y, const ALPHA_INT *indx, const ALPHA_Complex8 *x, const ALPHA_INT ns, const ALPHA_Complex8 val)
{
    ALPHA_INT ns4 = ((ns >> 2) << 2);
    ALPHA_INT i = 0;
    ALPHA_Complex8 tmp[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
#if defined(CMLA)
    float32x2_t v_tmp0 = vdup_n_f32(0), v_tmp1 = vdup_n_f32(0), v_tmp2 = vdup_n_f32(0), v_tmp3 = vdup_n_f32(0);
    float32x2_t v_x0, v_x1, v_x2, v_x3;
    float32x2_t v_y0, v_y1, v_y2, v_y3;
    float32x2_t v_val = vld1_f32((float *)&val);
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
        v_x0 = vld1_f32((float *)(x + i));
        v_x1 = vld1_f32((float *)(x + i + 1));
        v_x2 = vld1_f32((float *)(x + i + 2));
        v_x3 = vld1_f32((float *)(x + i + 3));

        v_y0 = vld1_f32((float *)(y + indx[i]));
        v_y1 = vld1_f32((float *)(y + indx[i + 1]));
        v_y2 = vld1_f32((float *)(y + indx[i + 2]));
        v_y3 = vld1_f32((float *)(y + indx[i + 3]));

        (v_y0) = vcmla_f32((v_y0), (v_x0), (v_val));
        (v_y1) = vcmla_f32((v_y1), (v_x1), (v_val));
        (v_y2) = vcmla_f32((v_y2), (v_x2), (v_val));
        (v_y3) = vcmla_f32((v_y3), (v_x3), (v_val));

        (v_y0) = vcmla_rot90_f32((v_y0), (v_x0), (v_val));
        (v_y1) = vcmla_rot90_f32((v_y1), (v_x1), (v_val));
        (v_y2) = vcmla_rot90_f32((v_y2), (v_x2), (v_val));
        (v_y3) = vcmla_rot90_f32((v_y3), (v_x3), (v_val));

        vst1_f32((float *)(y + indx[i]), v_y0);
        vst1_f32((float *)(y + indx[i + 1]), v_y1);
        vst1_f32((float *)(y + indx[i + 2]), v_y2);
        vst1_f32((float *)(y + indx[i + 3]), v_y3);
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
static inline void vec_fmai_conj_c(ALPHA_Complex8 *y, const ALPHA_INT *indx, const ALPHA_Complex8 *x, const ALPHA_INT ns, const ALPHA_Complex8 val)
{
    ALPHA_INT ns4 = ((ns >> 2) << 2);
    ALPHA_INT i = 0;
    ALPHA_Complex8 tmp[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
#if defined(CMLA)
    float32x2_t v_tmp0 = vdup_n_f32(0), v_tmp1 = vdup_n_f32(0), v_tmp2 = vdup_n_f32(0), v_tmp3 = vdup_n_f32(0);
    float32x2_t v_x0, v_x1, v_x2, v_x3;
    float32x2_t v_y0, v_y1, v_y2, v_y3;
    float32x2_t v_val = vld1_f32((float *)&val);
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
        v_x0 = vld1_f32((float *)(x + i));
        v_x1 = vld1_f32((float *)(x + i + 1));
        v_x2 = vld1_f32((float *)(x + i + 2));
        v_x3 = vld1_f32((float *)(x + i + 3));

        v_y0 = vld1_f32((float *)(y + indx[i]));
        v_y1 = vld1_f32((float *)(y + indx[i + 1]));
        v_y2 = vld1_f32((float *)(y + indx[i + 2]));
        v_y3 = vld1_f32((float *)(y + indx[i + 3]));

        (v_y0) = vcmla_f32((v_y0), (v_x0), (v_val));
        (v_y1) = vcmla_f32((v_y1), (v_x1), (v_val));
        (v_y2) = vcmla_f32((v_y2), (v_x2), (v_val));
        (v_y3) = vcmla_f32((v_y3), (v_x3), (v_val));

        (v_y0) = vcmla_rot270_f32((v_y0), (v_x0), (v_val));
        (v_y1) = vcmla_rot270_f32((v_y1), (v_x1), (v_val));
        (v_y2) = vcmla_rot270_f32((v_y2), (v_x2), (v_val));
        (v_y3) = vcmla_rot270_f32((v_y3), (v_x3), (v_val));

        vst1_f32((float *)(y + indx[i]), v_y0);
        vst1_f32((float *)(y + indx[i + 1]), v_y1);
        vst1_f32((float *)(y + indx[i + 2]), v_y2);
        vst1_f32((float *)(y + indx[i + 3]), v_y3);
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

static inline void vec_fma2_c(ALPHA_Complex8 *y, const ALPHA_Complex8 *x, const ALPHA_Complex8 val, const ALPHA_INT len)
{
    ALPHA_INT ns8 = ((len >> 3) << 3);
    ALPHA_INT i = 0;
#if defined(CMLA)
    float32x4_t v_tmp0 = vdupq_n_f32(0), v_tmp1 = vdupq_n_f32(0), v_tmp2 = vdupq_n_f32(0), v_tmp3 = vdupq_n_f32(0);
    float32x4_t v_x0, v_x1, v_x2, v_x3;
    float32x4_t v_y0, v_y1, v_y2, v_y3;
    float64x2_t v_val0 = vld1q_dup_f64((void *)&val);
    float32x4_t v_val = vreinterpretq_f32_f64(v_val0);
#endif

    for (i = 0; i < ns8; i += 8)
    {

#if defined(CMLA)
        v_x0 = vld1q_f32((float *)(x + i));
        v_x1 = vld1q_f32((float *)(x + i + 2));
        v_x2 = vld1q_f32((float *)(x + i + 4));
        v_x3 = vld1q_f32((float *)(x + i + 6));

        v_y0 = vld1q_f32((float *)(y + i));
        v_y1 = vld1q_f32((float *)(y + i + 2));
        v_y2 = vld1q_f32((float *)(y + i + 4));
        v_y3 = vld1q_f32((float *)(y + i + 6));

        (v_y0) = vcmlaq_f32((v_y0), (v_x0), (v_val));
        (v_y1) = vcmlaq_f32((v_y1), (v_x1), (v_val));
        (v_y2) = vcmlaq_f32((v_y2), (v_x2), (v_val));
        (v_y3) = vcmlaq_f32((v_y3), (v_x3), (v_val));

        (v_y0) = vcmlaq_rot90_f32((v_y0), (v_x0), (v_val));
        (v_y1) = vcmlaq_rot90_f32((v_y1), (v_x1), (v_val));
        (v_y2) = vcmlaq_rot90_f32((v_y2), (v_x2), (v_val));
        (v_y3) = vcmlaq_rot90_f32((v_y3), (v_x3), (v_val));

        vst1q_f32((float *)(y + i), v_y0);
        vst1q_f32((float *)(y + i + 2), v_y1);
        vst1q_f32((float *)(y + i + 4), v_y2);
        vst1q_f32((float *)(y + i + 6), v_y3);
#else
        cmp_madde(y[i], x[i], val);
        cmp_madde(y[i + 1], x[i + 1], val);
        cmp_madde(y[i + 2], x[i + 2], val);
        cmp_madde(y[i + 3], x[i + 3], val);

        cmp_madde(y[i + 4], x[i + 4], val);
        cmp_madde(y[i + 5], x[i + 5], val);
        cmp_madde(y[i + 6], x[i + 6], val);
        cmp_madde(y[i + 7], x[i + 7], val);
#endif
    }
    for (; i < len; ++i)
    {
        cmp_madde(y[i], x[i], val);
    }
}