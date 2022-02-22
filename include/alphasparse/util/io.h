#pragma once

/**
 * @brief header for file read and write utils
 */

#include "../spdef.h"
#include "../types.h"
#include <stdlib.h>
#include <stdio.h>

#define BUFFER_SIZE 1024

FILE *alpha_open(const char * filename, const char * modes);
void alpha_close(FILE *stream);

void result_write(const char *path, const size_t ele_num, size_t ele_size, const void *data);

void alpha_read_coo(const char *file, ALPHA_INT *m_p, ALPHA_INT *n_p, ALPHA_INT *nnz_p, ALPHA_INT **row_index, ALPHA_INT **col_index, float **values);
void alpha_read_coo_d(const char *file, ALPHA_INT *m_p, ALPHA_INT *n_p, ALPHA_INT *nnz_p, ALPHA_INT **row_index, ALPHA_INT **col_index, double **values);
void alpha_read_coo_c(const char *file, ALPHA_INT *m_p, ALPHA_INT *n_p, ALPHA_INT *nnz_p, ALPHA_INT **row_index, ALPHA_INT **col_index, ALPHA_Complex8 **values);
void alpha_read_coo_z(const char *file, ALPHA_INT *m_p, ALPHA_INT *n_p, ALPHA_INT *nnz_p, ALPHA_INT **row_index, ALPHA_INT **col_index, ALPHA_Complex16 **values);

#ifdef __MKL__

#include <mkl.h>
void mkl_read_coo(const char *file, MKL_INT *m_p, MKL_INT *n_p, MKL_INT *nnz_p, MKL_INT **row_index, MKL_INT **col_index, float **values);
void mkl_read_coo_d(const char *file, MKL_INT *m_p, MKL_INT *n_p, MKL_INT *nnz_p, MKL_INT **row_index, MKL_INT **col_index, double **values);
void mkl_read_coo_c(const char *file, MKL_INT *m_p, MKL_INT *n_p, MKL_INT *nnz_p, MKL_INT **row_index, MKL_INT **col_index, MKL_Complex8 **values);
void mkl_read_coo_z(const char *file, MKL_INT *m_p, MKL_INT *n_p, MKL_INT *nnz_p, MKL_INT **row_index, MKL_INT **col_index, MKL_Complex16 **values);
#endif 