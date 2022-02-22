#pragma once

/**
 * @brief header for parameter check utils
 */

#include <stdlib.h>
#include "../spdef.h"
#include "../types.h"
#include "../spmat.h"
#include <stdbool.h>

float CalEpisilon_s();
double CalEpisilon_d();

double InfiniteNorm_d(const ALPHA_INT n, const double *xa, const double *xb);
double GeNorm1_d(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * col_index, const double *val);
double TrSyNorm1_d(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const double *val, struct alpha_matrix_descr descr);
double DiNorm1_d(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const double *val, struct alpha_matrix_descr descr);

double DesNorm1_d(ALPHA_INT rows, ALPHA_INT cols, const double *val, ALPHA_INT ldv, alphasparse_layout_t layout);
double DesDiffNorm1_d(const ALPHA_INT rows, const ALPHA_INT cols, const double *xa, const ALPHA_INT lda, const double *xb, const ALPHA_INT ldb, alphasparse_layout_t layout);

float InfiniteNorm_s(const ALPHA_INT n, const float *xa, const float *xb);
float GeNorm1_s(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * col_index, const float *val);
float TrSyNorm1_s(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const float *val, struct alpha_matrix_descr descr);
float DiNorm1_s(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const float *val, struct alpha_matrix_descr descr);

float DesNorm1_s(ALPHA_INT rows, ALPHA_INT cols, const float *val, ALPHA_INT ldv, alphasparse_layout_t layout);
float DesDiffNorm1_s(const ALPHA_INT rows, const ALPHA_INT cols, const float *xa, const ALPHA_INT lda, const float *xb, const ALPHA_INT ldb, alphasparse_layout_t layout);

float InfiniteNorm_c(const ALPHA_INT n, const ALPHA_Complex8 *xa, const ALPHA_Complex8 *xb);
float GeNorm1_c(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * col_index, const ALPHA_Complex8 *val);
float TrSyNorm1_c(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const ALPHA_Complex8 *val, struct alpha_matrix_descr descr);
float DiNorm1_c(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const ALPHA_Complex8 *val, struct alpha_matrix_descr descr);

float DesNorm1_c(ALPHA_INT rows, ALPHA_INT cols, const ALPHA_Complex8 *val, ALPHA_INT ldv, alphasparse_layout_t layout);
float DesDiffNorm1_c(const ALPHA_INT rows, const ALPHA_INT cols, const ALPHA_Complex8 *xa, const ALPHA_INT lda, const ALPHA_Complex8 *xb, const ALPHA_INT ldb, alphasparse_layout_t layout);

double InfiniteNorm_z(const ALPHA_INT n, const ALPHA_Complex16 *xa, const ALPHA_Complex16 *xb);
double GeNorm1_z(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * col_index, const ALPHA_Complex16 *val);
double TrSyNorm1_z(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const ALPHA_Complex16 *val, struct alpha_matrix_descr descr);
double DiNorm1_z(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const ALPHA_Complex16 *val, struct alpha_matrix_descr descr);

double DesNorm1_z(ALPHA_INT rows, ALPHA_INT cols, const ALPHA_Complex16 *val, ALPHA_INT ldv, alphasparse_layout_t layout);
double DesDiffNorm1_z(const ALPHA_INT rows, const ALPHA_INT cols, const ALPHA_Complex16 *xa, const ALPHA_INT lda, const ALPHA_Complex16 *xb, const ALPHA_INT ldb, alphasparse_layout_t layout);

