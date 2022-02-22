#pragma once
#include <stdint.h>
#include "alphasparse/types.h"
#include "alphasparse/spmat.h"

bool LUdecompose_s(ALPHA_INT row, ALPHA_INT col, float *x, float *y);
bool LUdecompose_d(ALPHA_INT row, ALPHA_INT col, double *x, double *y);
bool LUdecompose_c(ALPHA_INT row, ALPHA_INT col, ALPHA_Complex8 *x, ALPHA_Complex8 *y);
bool LUdecompose_z(ALPHA_INT row, ALPHA_INT col, ALPHA_Complex16 *x, ALPHA_Complex16 *y);


#ifndef COMPLEX
#ifndef DOUBLE

#define LUdecompose LUdecompose_s
#else

#define LUdecompose LUdecompose_d

#endif
#else
#ifndef DOUBLE

#define LUdecompose LUdecompose_c

#else

#define LUdecompose LUdecompose_z


#endif
#endif