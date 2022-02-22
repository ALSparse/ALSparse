#pragma once

#include "types.h"
#include "complex_compute.h"
#include "real_compute.h"

#ifndef COMPLEX

#define alpha_setone real_setone
#define alpha_setzero real_setzero
#define alpha_iszero real_iszero
#define alpha_isone real_isone
#define alpha_add real_add
#define alpha_sub real_sub
#define alpha_mul real_mul
#define alpha_div real_div
#define alpha_madd real_madd
#define alpha_msub real_msub

#define alpha_adde real_adde
#define alpha_sube real_sube
#define alpha_mule real_mule
#define alpha_dive real_dive
#define alpha_madde real_madde
#define alpha_msube real_msube

#define alpha_copy real_copy
#define alpha_cross_entropy real_cross_entropy

#else

#define alpha_setone cmp_setone
#define alpha_setzero cmp_setzero
#define alpha_iszero cmp_iszero
#define alpha_isone cmp_isone
#define alpha_add cmp_add
#define alpha_sub cmp_sub
#define alpha_mul cmp_mul
#define alpha_mul_2c cmp_mul_2c
#define alpha_mul_3c cmp_mul_3c
#define alpha_div cmp_div
#define alpha_div_3c cmp_div_3c
#define alpha_madd cmp_madd
#define alpha_madd_2c cmp_madd_2c
#define alpha_madd_3c cmp_madd_3c
#define alpha_msub cmp_msub

#define alpha_adde cmp_adde
#define alpha_sube cmp_sube
#define alpha_mule cmp_mule
#define alpha_mule_2c cmp_mule_2c
#define alpha_mule_3c cmp_mule_3c
#define alpha_dive cmp_dive
#define alpha_madde cmp_madde
#define alpha_madde_2c cmp_madde_2c
#define alpha_madde_3c cmp_madde_3c
#define alpha_msube cmp_msube
#define alpha_msube_2c cmp_msube_2c
#define alpha_conj cmp_conj

#define alpha_copy cmp_copy
#define alpha_cross_entropy cmp_cross_entropy

#endif
