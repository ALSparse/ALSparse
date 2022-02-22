#pragma once

#include "../spdef.h"
#include "../types.h"

alphasparse_status_t axpy_c_(const ALPHA_INT nz, const ALPHA_Complex8 a, const ALPHA_Complex8 *x, const ALPHA_INT *indx, ALPHA_Complex8 *y);
alphasparse_status_t gthr_c_(const ALPHA_INT nz, const ALPHA_Complex8 *y, ALPHA_Complex8 *x, const ALPHA_INT *indx);
alphasparse_status_t gthrz_c_(const ALPHA_INT nz, ALPHA_Complex8 *y, ALPHA_Complex8 *x, const ALPHA_INT *indx);
alphasparse_status_t sctr_c_(const ALPHA_INT nz, const ALPHA_Complex8 *x, const ALPHA_INT *indx, ALPHA_Complex8 *y);
void dotui_c_sub(const ALPHA_INT nz, const ALPHA_Complex8 *x, const ALPHA_INT *indx, const ALPHA_Complex8 *y, ALPHA_Complex8 *dutui);
void dotci_c_sub(const ALPHA_INT nz, const ALPHA_Complex8 *x, const ALPHA_INT *indx, const ALPHA_Complex8 *y, ALPHA_Complex8 *dutci);



