#pragma once

#include "../types.h"
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#include "../compute.h"
#define vec_doti vec_doti_z
#define VEC_DOTADD_4 VEC_DOTADD_Z4
#define VEC_DOTSUB_4 VEC_DOTSUB_Z4
#define VEC_DOTSUB_4_CONJ1 VEC_DOTSUB_Z4_CONJ1
#define vec_doti_conj vec_doti_conj_z
// sum -= inner_product(a[4],b[4])

#ifdef __aarch64__
#define VEC_DOTADD_Z4(a, b, sum)                                         \
    do                                                                   \
    {                                                                    \
        float64x2x2_t v_a = vld2q_f64((double *)(a));                    \
        float64x2x2_t v_b = vld2q_f64((double *)(b));                    \
        float64x2x2_t v_a_2 = vld2q_f64((double *)((a) + 2));            \
        float64x2x2_t v_b_2 = vld2q_f64((double *)((b) + 2));            \
        float64x2_t ar_br = vmulq_f64(v_a.val[0], v_b.val[0]);           \
        float64x2_t ar_bi = vmulq_f64(v_a.val[0], v_b.val[1]);           \
        float64x2_t ar_br_2 = vmulq_f64(v_a_2.val[0], v_b_2.val[0]);     \
        float64x2_t ar_bi_2 = vmulq_f64(v_a_2.val[0], v_b_2.val[1]);     \
        float64x2x2_t z_v_0, z_v_2;                                      \
        z_v_0.val[0] = vfmsq_f64(ar_br, v_a.val[1], v_b.val[1]);         \
        z_v_0.val[1] = vfmaq_f64(ar_bi, v_a.val[1], v_b.val[0]);         \
        z_v_2.val[0] = vfmsq_f64(ar_br_2, v_a_2.val[1], v_b_2.val[1]);   \
        z_v_2.val[1] = vfmaq_f64(ar_bi_2, v_a_2.val[1], v_b_2.val[0]);   \
        (sum).real += vaddvq_f64(vaddq_f64(z_v_0.val[0], z_v_2.val[0])); \
        (sum).imag += vaddvq_f64(vaddq_f64(z_v_0.val[1], z_v_2.val[1])); \
    } while (0)
#else
#define VEC_DOTADD_Z4(a, b, sum)                                                                       \
    do                                                                                                 \
    {                                                                                                  \
        for (int32_t __i = 0; __i < 4; __i++)                                                          \
        {                                                                                              \
            double _B = (a)[__i].real * (b)[__i].imag;                                                 \
            double _C = (a)[__i].imag * (b)[__i].real;                                                 \
            (sum).real += ((a)[__i].real + (a)[__i].imag) * ((b)[__i].real - (b)[__i].imag) + _B - _C; \
            (sum).imag += _B + _C;                                                                     \
        }                                                                                              \
    } while (0)
#endif

// sum -= inner_product(a[4],b[4])
#ifdef __aarch64__
#define VEC_DOTSUB_Z4(a, b, sum)                                         \
    do                                                                   \
    {                                                                    \
        float64x2x2_t v_a = vld2q_f64((double *)(a));                    \
        float64x2x2_t v_b = vld2q_f64((double *)(b));                    \
        float64x2x2_t v_a_2 = vld2q_f64((double *)((a) + 2));            \
        float64x2x2_t v_b_2 = vld2q_f64((double *)((b) + 2));            \
        float64x2_t ar_br = vmulq_f64(v_a.val[0], v_b.val[0]);           \
        float64x2_t ar_bi = vmulq_f64(v_a.val[0], v_b.val[1]);           \
        float64x2_t ar_br_2 = vmulq_f64(v_a_2.val[0], v_b_2.val[0]);     \
        float64x2_t ar_bi_2 = vmulq_f64(v_a_2.val[0], v_b_2.val[1]);     \
        float64x2x2_t z_v_0, z_v_2;                                      \
        z_v_0.val[0] = vfmsq_f64(ar_br, v_a.val[1], v_b.val[1]);         \
        z_v_0.val[1] = vfmaq_f64(ar_bi, v_a.val[1], v_b.val[0]);         \
        z_v_2.val[0] = vfmsq_f64(ar_br_2, v_a_2.val[1], v_b_2.val[1]);   \
        z_v_2.val[1] = vfmaq_f64(ar_bi_2, v_a_2.val[1], v_b_2.val[0]);   \
        (sum).real -= vaddvq_f64(vaddq_f64(z_v_0.val[0], z_v_2.val[0])); \
        (sum).imag -= vaddvq_f64(vaddq_f64(z_v_0.val[1], z_v_2.val[1])); \
    } while (0)
#else
#define VEC_DOTSUB_Z4(a, b, sum)                                                                       \
    do                                                                                                 \
    {                                                                                                  \
        for (int32_t __i = 0; __i < 4; __i++)                                                          \
        {                                                                                              \
            double _B = (a)[__i].real * (b)[__i].imag;                                                 \
            double _C = (a)[__i].imag * (b)[__i].real;                                                 \
            (sum).real -= ((a)[__i].real + (a)[__i].imag) * ((b)[__i].real - (b)[__i].imag) + _B - _C; \
            (sum).imag -= _B + _C;                                                                     \
        }                                                                                              \
    } while (0)
#endif
#ifdef __aarch64__
#define VEC_DOTSUB_Z4_CONJ1(a, b, sum)                                   \
    do                                                                   \
    {                                                                    \
        float64x2x2_t v_a = vld2q_f64((double *)(a));                    \
        float64x2x2_t v_b = vld2q_f64((double *)(b));                    \
        float64x2x2_t v_a_2 = vld2q_f64((double *)((a) + 2));            \
        float64x2x2_t v_b_2 = vld2q_f64((double *)((b) + 2));            \
        float64x2_t ar_br = vmulq_f64(v_a.val[0], v_b.val[0]);           \
        float64x2_t ar_bi = vmulq_f64(v_a.val[0], v_b.val[1]);           \
        float64x2_t ar_br_2 = vmulq_f64(v_a_2.val[0], v_b_2.val[0]);     \
        float64x2_t ar_bi_2 = vmulq_f64(v_a_2.val[0], v_b_2.val[1]);     \
        float64x2x2_t z_v_0, z_v_2;                                      \
        z_v_0.val[0] = vfmaq_f64(ar_br, v_a.val[1], v_b.val[1]);         \
        z_v_0.val[1] = vfmsq_f64(ar_bi, v_a.val[1], v_b.val[0]);         \
        z_v_2.val[0] = vfmaq_f64(ar_br_2, v_a_2.val[1], v_b_2.val[1]);   \
        z_v_2.val[1] = vfmsq_f64(ar_bi_2, v_a_2.val[1], v_b_2.val[0]);   \
        (sum).real -= vaddvq_f64(vaddq_f64(z_v_0.val[0], z_v_2.val[0])); \
        (sum).imag -= vaddvq_f64(vaddq_f64(z_v_0.val[1], z_v_2.val[1])); \
    } while (0)
#else
#define VEC_DOTSUB_Z4_CONJ1(a, b, sum)                                                                 \
    do                                                                                                 \
    {                                                                                                  \
        for (int32_t __i = 0; __i < 4; __i++)                                                          \
        {                                                                                              \
            double _B = (a)[__i].real * (b)[__i].imag;                                                 \
            double _C = -(a)[__i].imag * (b)[__i].real;                                                \
            (sum).real -= ((a)[__i].real - (a)[__i].imag) * ((b)[__i].real - (b)[__i].imag) + _B - _C; \
            (sum).imag -= _B + _C;                                                                     \
        }                                                                                              \
    } while (0)
#endif

static inline ALPHA_Complex16 vec_doti_z(const ALPHA_INT ns, const ALPHA_Complex16 *x, const ALPHA_INT *indx, const ALPHA_Complex16 *y)

{
    ALPHA_INT ns4 = ((ns >> 2) << 2);
    ALPHA_INT i = 0;
    ALPHA_Complex16 tmp[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
#if defined(CMLA)
    float64x2_t v_tmp0 = vdupq_n_f64(0), v_tmp1 = vdupq_n_f64(0), v_tmp2 = vdupq_n_f64(0), v_tmp3 = vdupq_n_f64(0);
    float64x2_t v_x0, v_x1, v_x2, v_x3;
    float64x2_t v_y0, v_y1, v_y2, v_y3;
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

        (v_tmp0) = vcmlaq_f64((v_tmp0), (v_x0), (v_y0));
        (v_tmp1) = vcmlaq_f64((v_tmp1), (v_x1), (v_y1));
        (v_tmp2) = vcmlaq_f64((v_tmp2), (v_x2), (v_y2));
        (v_tmp3) = vcmlaq_f64((v_tmp3), (v_x3), (v_y3));

        (v_tmp0) = vcmlaq_rot90_f64((v_tmp0), (v_x0), (v_y0));
        (v_tmp1) = vcmlaq_rot90_f64((v_tmp1), (v_x1), (v_y1));
        (v_tmp2) = vcmlaq_rot90_f64((v_tmp2), (v_x2), (v_y2));
        (v_tmp3) = vcmlaq_rot90_f64((v_tmp3), (v_x3), (v_y3));
#else
        cmp_madde(tmp[0], x[i], y[indx[i]]);
        cmp_madde(tmp[1], x[i + 1], y[indx[i + 1]]);
        cmp_madde(tmp[2], x[i + 2], y[indx[i + 2]]);
        cmp_madde(tmp[3], x[i + 3], y[indx[i + 3]]);
#endif
    }
    for (; i < ns; ++i)
    {
        cmp_madde(tmp[0], x[i], y[indx[i]]);
    }

#if defined(CMLA)
    ALPHA_Complex16 CT[4];
    vst1q_f64((double *)(CT), v_tmp0);
    vst1q_f64((double *)(CT + 1), v_tmp1);
    vst1q_f64((double *)(CT + 2), v_tmp2);
    vst1q_f64((double *)(CT + 3), v_tmp3);
    cmp_adde(tmp[0], CT[0]);
    cmp_adde(tmp[1], CT[1]);
    cmp_adde(tmp[2], CT[2]);
    cmp_adde(tmp[3], CT[3]);
#endif
    cmp_adde(tmp[0], tmp[1]);
    cmp_adde(tmp[2], tmp[3]);
    cmp_adde(tmp[0], tmp[2]);
    return tmp[0];
}
static inline ALPHA_Complex16 vec_doti_conj_z(const ALPHA_INT ns, const ALPHA_Complex16 *x, const ALPHA_INT *indx, const ALPHA_Complex16 *y)
{
    ALPHA_INT ns4 = ((ns >> 2) << 2);
    ALPHA_INT i = 0;
    ALPHA_Complex16 tmp[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
#if defined(CMLA)
    float64x2_t v_tmp0 = vdupq_n_f64(0), v_tmp1 = vdupq_n_f64(0), v_tmp2 = vdupq_n_f64(0), v_tmp3 = vdupq_n_f64(0);
    float64x2_t v_x0, v_x1, v_x2, v_x3;
    float64x2_t v_y0, v_y1, v_y2, v_y3;
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

        (v_tmp0) = vcmlaq_f64((v_tmp0), (v_x0), (v_y0));
        (v_tmp1) = vcmlaq_f64((v_tmp1), (v_x1), (v_y1));
        (v_tmp2) = vcmlaq_f64((v_tmp2), (v_x2), (v_y2));
        (v_tmp3) = vcmlaq_f64((v_tmp3), (v_x3), (v_y3));

        (v_tmp0) = vcmlaq_rot270_f64((v_tmp0), (v_x0), (v_y0));
        (v_tmp1) = vcmlaq_rot270_f64((v_tmp1), (v_x1), (v_y1));
        (v_tmp2) = vcmlaq_rot270_f64((v_tmp2), (v_x2), (v_y2));
        (v_tmp3) = vcmlaq_rot270_f64((v_tmp3), (v_x3), (v_y3));
#else
        cmp_madde_2c(tmp[0], x[i], y[indx[i]]);
        cmp_madde_2c(tmp[1], x[i + 1], y[indx[i + 1]]);
        cmp_madde_2c(tmp[2], x[i + 2], y[indx[i + 2]]);
        cmp_madde_2c(tmp[3], x[i + 3], y[indx[i + 3]]);
#endif
    }
    for (; i < ns; ++i)
    {
        cmp_madde_2c(tmp[0], x[i], y[indx[i]]);
    }

#if defined(CMLA)
    ALPHA_Complex16 CT[4];
    vst1q_f64((double *)(CT), v_tmp0);
    vst1q_f64((double *)(CT + 1), v_tmp1);
    vst1q_f64((double *)(CT + 2), v_tmp2);
    vst1q_f64((double *)(CT + 3), v_tmp3);
    cmp_adde(tmp[0], CT[0]);
    cmp_adde(tmp[1], CT[1]);
    cmp_adde(tmp[2], CT[2]);
    cmp_adde(tmp[3], CT[3]);
#endif
    cmp_adde(tmp[0], tmp[1]);
    cmp_adde(tmp[2], tmp[3]);
    cmp_adde(tmp[0], tmp[2]);
    return tmp[0];
}
