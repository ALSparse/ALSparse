#pragma once

#include "../types.h"
#include "thread.h"

void pack_matrix_col2row_s(const ALPHA_INT rowX, const ALPHA_INT colX, const float *X, const ALPHA_INT ldX, float * Y, ALPHA_INT ldY);
void pack_matrix_col2row_d(const ALPHA_INT rowX, const ALPHA_INT colX, const double *X, const ALPHA_INT ldX, double * Y, ALPHA_INT ldY);
void pack_matrix_col2row_c(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex8 *X, const ALPHA_INT ldX, ALPHA_Complex8 * Y, ALPHA_INT ldY);
void pack_matrix_col2row_z(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex16 *X, const ALPHA_INT ldX, ALPHA_Complex16 * Y, ALPHA_INT ldY);

void pack_matrix_row2col_s(const ALPHA_INT rowX, const ALPHA_INT colX, const float *X, const ALPHA_INT ldX, float * Y, ALPHA_INT ldY);
void pack_matrix_row2col_d(const ALPHA_INT rowX, const ALPHA_INT colX, const double *X, const ALPHA_INT ldX, double * Y, ALPHA_INT ldY);
void pack_matrix_row2col_c(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex8 *X, const ALPHA_INT ldX, ALPHA_Complex8 * Y, ALPHA_INT ldY);
void pack_matrix_row2col_z(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex16 *X, const ALPHA_INT ldX, ALPHA_Complex16 * Y, ALPHA_INT ldY);


#ifdef S
#define pack_matrix_col2row pack_matrix_col2row_s
#define pack_matrix_row2col pack_matrix_row2col_s
#endif
#ifdef D
#define pack_matrix_col2row pack_matrix_col2row_d
#define pack_matrix_row2col pack_matrix_row2col_d
#endif
#ifdef C
#define pack_matrix_col2row pack_matrix_col2row_c
#define pack_matrix_row2col pack_matrix_row2col_c
#endif
#ifdef Z
#define pack_matrix_col2row pack_matrix_col2row_z
#define pack_matrix_row2col pack_matrix_row2col_z
#endif












