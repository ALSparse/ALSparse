#pragma once

/**
 * @brief header for parameter check utils
 */

#include <stdlib.h>
#include "../spdef.h"
#include "../types.h"
#include "../spmat.h"
#include <stdbool.h>

#define check_null_return(pointer, error) \
    if (((void *)pointer) == NULL)        \
    {                                     \
        return error;                     \
    }

#define check_return(expr, error) \
    if (expr)                     \
    {                             \
        return error;             \
    }

// [min max)
#define check_between_return(num, min, max, error) \
    if ((num) >= (max) || (num) < (min))           \
    {                                              \
        return error;                              \
    }

#define check_error_return(func)                  \
    {                                             \
        alphasparse_status_t _status = func;       \
        if (_status != ALPHA_SPARSE_STATUS_SUCCESS) \
            return _status;                       \
    }

#define THRESH 50.0f

bool check_equal_row_col(const alphasparse_matrix_t A);
bool check_equal_colA_rowB(const alphasparse_matrix_t A, const alphasparse_matrix_t B, const alphasparse_operation_t transA);

// success 0 error -1
int check_s(const float *answer_data, size_t answer_size, const float *result_data, size_t result_size);
int check_d(const double *answer_data, size_t answer_size, const double *result_data, size_t result_size);
int check_c(const ALPHA_Complex8 *answer_data, size_t answer_size, const ALPHA_Complex8 *result_data, size_t result_size);
int check_z(const ALPHA_Complex16 *answer_data, size_t answer_size, const ALPHA_Complex16 *result_data, size_t result_size);

int check_s_l2(const float *answer_data, size_t answer_size, const float *result_data, size_t result_size, const float *x, const float *y, const float alpha, const float beta, int argc, const char *argv[]);
int check_s_l3(const float *answer_data, const ALPHA_INT ldans, size_t answer_size, const float *result_data, const ALPHA_INT ldres, size_t result_size, const ALPHA_INT *res_col_indx, const float *x, const ALPHA_INT ldx, const float *y, const ALPHA_INT ldy, const float alpha, const float beta, int argc, const char *argv[]);
int check_d_l2(const double *answer_data, size_t answer_size, const double *result_data, size_t result_size, const double *x, const double *y, const double alpha, const double beta, int argc, const char *argv[]);
int check_d_l3(const double *answer_data, const ALPHA_INT ldans, size_t answer_size, const double *result_data, const ALPHA_INT ldres, size_t result_size, const ALPHA_INT *res_col_indx, const double *x, const ALPHA_INT ldx, const double *y, const ALPHA_INT ldy, const double alpha, const double beta, int argc, const char *argv[]);
int check_c_l2(const ALPHA_Complex8 *answer_data, size_t answer_size, const ALPHA_Complex8 *result_data, size_t result_size, const ALPHA_Complex8 *x, const ALPHA_Complex8 *y, const ALPHA_Complex8 alpha, const ALPHA_Complex8 beta, int argc, const char *argv[]);
int check_c_l3(const ALPHA_Complex8 *answer_data, const ALPHA_INT ldans, size_t answer_size, const ALPHA_Complex8 *result_data, const ALPHA_INT ldres, size_t result_size, const ALPHA_INT *res_col_indx, const ALPHA_Complex8 *x, const ALPHA_INT ldx, const ALPHA_Complex8 *y, const ALPHA_INT ldy, const ALPHA_Complex8 alpha, const ALPHA_Complex8 beta, int argc, const char *argv[]);
int check_z_l2(const ALPHA_Complex16 *answer_data, size_t answer_size, const ALPHA_Complex16 *result_data, size_t result_size, const ALPHA_Complex16 *x, const ALPHA_Complex16 *y, const ALPHA_Complex16 alpha, const ALPHA_Complex16 beta, int argc, const char *argv[]);
int check_z_l3(const ALPHA_Complex16 *answer_data, const ALPHA_INT ldans, size_t answer_size, const ALPHA_Complex16 *result_data, const ALPHA_INT ldres, size_t result_size, const ALPHA_INT *res_col_indx, const ALPHA_Complex16 *x, const ALPHA_INT ldx, const ALPHA_Complex16 *y, const ALPHA_INT ldy, const ALPHA_Complex16 alpha, const ALPHA_Complex16 beta, int argc, const char *argv[]);
