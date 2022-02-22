#pragma once

#include "../types.h"
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#include "../compute.h"
#define vec_doti vec_doti_d
#define VEC_DOTADD_4 VEC_DOTADD_D4
#define VEC_DOTSUB_4 VEC_DOTSUB_D4

#ifdef __aarch64__
#define VEC_DOTADD_D4(a, b, sum)                            \
    do                                                      \
    {                                                       \
        float64x2_t v_a = vld1q_f64((double *)(a));         \
        float64x2_t v_a_2 = vld1q_f64((double *)((a) + 2)); \
        float64x2_t v_b = vld1q_f64((double *)(b));         \
        float64x2_t v_b_2 = vld1q_f64((double *)((b) + 2)); \
        float64x2_t prod = vmulq_f64(v_a, v_b);             \
        float64x2_t prod_2 = vfmaq_f64(prod, v_a_2, v_b_2); \
        (sum) += vaddvq_f64(prod_2);                        \
    } while (0)
#else
#define VEC_DOTADD_D4(a, b, sum)  \
    do                            \
    {                             \
        (sum) += (a)[0] * (b)[0]; \
        (sum) += (a)[1] * (b)[1]; \
        (sum) += (a)[2] * (b)[2]; \
        (sum) += (a)[3] * (b)[3]; \
    } while (0)
#endif

#ifdef __aarch64__
#define VEC_DOTSUB_D4(a, b, sum)                            \
    do                                                      \
    {                                                       \
        float64x2_t v_a = vld1q_f64((double *)(a));         \
        float64x2_t v_a_2 = vld1q_f64((double *)((a) + 2)); \
        float64x2_t v_b = vld1q_f64((double *)(b));         \
        float64x2_t v_b_2 = vld1q_f64((double *)((b) + 2)); \
        float64x2_t prod = vmulq_f64(v_a, v_b);             \
        float64x2_t prod_2 = vfmaq_f64(prod, v_a_2, v_b_2); \
        (sum) -= vaddvq_f64(prod_2);                        \
    } while (0)
#else
#define VEC_DOTSUB_D4(a, b, sum)  \
    do                            \
    {                             \
        (sum) -= (a)[0] * (b)[0]; \
        (sum) -= (a)[1] * (b)[1]; \
        (sum) -= (a)[2] * (b)[2]; \
        (sum) -= (a)[3] * (b)[3]; \
    } while (0)
#endif
static inline double vec_doti_d(const ALPHA_INT ns, const double *x, const ALPHA_INT *indx, const double *y)
{
    ALPHA_INT ns4 = ((ns >> 2) << 2);
    ALPHA_INT i = 0;
    double tmp[4] = {
        0.,
        0.,
        0.,
        0.,
    };
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
        real_madde(tmp[0], x[i], y[indx[i]]);
        real_madde(tmp[1], x[i + 1], y[indx[i + 1]]);
        real_madde(tmp[2], x[i + 2], y[indx[i + 2]]);
        real_madde(tmp[3], x[i + 3], y[indx[i + 3]]);
    }
    for (; i < ns; ++i)
    {
        real_madde(tmp[0], x[i], y[indx[i]]);
    }
    return (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
}