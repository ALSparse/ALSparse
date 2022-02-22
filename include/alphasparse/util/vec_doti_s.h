#pragma once

#include "../types.h"
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#include"../compute.h"
#define vec_doti vec_doti_s
#define VEC_DOTADD_4 VEC_DOTADD_S4
#define VEC_DOTSUB_4 VEC_DOTSUB_S4

#ifdef __aarch64__
#define VEC_DOTADD_S4(a, b, sum)                \
    do                                          \
    {                                           \
        float32x4_t v_a = vld1q_f32((a));       \
        float32x4_t v_b = vld1q_f32((b));       \
        float32x4_t prod = vmulq_f32(v_a, v_b); \
        (sum) += vaddvq_f32(prod);              \
    } while (0)
#else
#define VEC_DOTADD_S4(a, b, sum)  \
    do                            \
    {                             \
        (sum) += (a)[0] * (b)[0]; \
        (sum) += (a)[1] * (b)[1]; \
        (sum) += (a)[2] * (b)[2]; \
        (sum) += (a)[3] * (b)[3]; \
    } while (0)
#endif

// sum -= inner_product(a[4],b[4])
#ifdef __aarch64__
#define VEC_DOTSUB_S4(a, b, sum)                \
    do                                          \
    {                                           \
        float32x4_t v_a = vld1q_f32((a));       \
        float32x4_t v_b = vld1q_f32((b));       \
        float32x4_t prod = vmulq_f32(v_a, v_b); \
        (sum) -= vaddvq_f32(prod);              \
    } while (0)
#else
#define VEC_DOTSUB_S4(a, b, sum)  \
    do                            \
    {                             \
        (sum) -= (a)[0] * (b)[0]; \
        (sum) -= (a)[1] * (b)[1]; \
        (sum) -= (a)[2] * (b)[2]; \
        (sum) -= (a)[3] * (b)[3]; \
    } while (0)
#endif
static inline float vec_doti_s(const ALPHA_INT ns, const float *x, const ALPHA_INT *indx, const float *y)
{
    ALPHA_INT ns4 = ((ns >> 2) << 2);
    ALPHA_INT i = 0;
    __asm__ volatile(
        "prfm pldl3strm, [%[x]]\n\t"
        "prfm pldl3strm, [%[indx]]\n\t"
        :
        : [ x ] "r"(x), [ indx ] "r"(indx));
    float tmp[4] = {
        0.,
        0.,
        0.,
        0.,
    };
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
