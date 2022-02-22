#pragma once

#include "../spdef.h"
#include "../types.h"

alphasparse_status_t axpy_s_(const ALPHA_INT nz, const float a, const float *x, const ALPHA_INT *indx, float *y);
alphasparse_status_t gthr_s_(const ALPHA_INT nz, const float *y, float *x, const ALPHA_INT *indx);
alphasparse_status_t gthrz_s_(const ALPHA_INT nz, float *y, float *x, const ALPHA_INT *indx);
alphasparse_status_t rot_s_(const ALPHA_INT nz, float *x, const ALPHA_INT *indx, float *y, const float c, const float s);
alphasparse_status_t sctr_s_(const ALPHA_INT nz, const float *x, const ALPHA_INT *indx, float *y);
float doti_s_(const ALPHA_INT nz, const float *x, const ALPHA_INT *indx, const float *y);

