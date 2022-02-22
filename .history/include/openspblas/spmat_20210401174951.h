#pragma once

/**
 *  @brief header for the internal sparse matrix definitions
 */

#include "spdef.h"
#include "types.h"

#ifndef COMPLEX

#ifndef DOUBLE
#define OPENSPBLAS_SPMAT_COO spmat_coo_s_t
#define OPENSPBLAS_SPMAT_CSR spmat_csr_s_t
#define OPENSPBLAS_SPMAT_CSC spmat_csc_s_t
#define OPENSPBLAS_SPMAT_BSR spmat_bsr_s_t
#define OPENSPBLAS_SPMAT_DIA spmat_dia_s_t
#define OPENSPBLAS_SPMAT_SKY spmat_sky_s_t
#else
#define OPENSPBLAS_SPMAT_COO spmat_coo_d_t
#define OPENSPBLAS_SPMAT_CSR spmat_csr_d_t
#define OPENSPBLAS_SPMAT_CSC spmat_csc_d_t
#define OPENSPBLAS_SPMAT_BSR spmat_bsr_d_t
#define OPENSPBLAS_SPMAT_DIA spmat_dia_d_t
#define OPENSPBLAS_SPMAT_SKY spmat_sky_d_t
#endif

#else

#ifndef DOUBLE
#define OPENSPBLAS_SPMAT_COO spmat_coo_c_t
#define OPENSPBLAS_SPMAT_CSR spmat_csr_c_t
#define OPENSPBLAS_SPMAT_CSC spmat_csc_c_t
#define OPENSPBLAS_SPMAT_BSR spmat_bsr_c_t
#define OPENSPBLAS_SPMAT_DIA spmat_dia_c_t
#define OPENSPBLAS_SPMAT_SKY spmat_sky_c_t
#else
#define OPENSPBLAS_SPMAT_COO spmat_coo_z_t
#define OPENSPBLAS_SPMAT_CSR spmat_csr_z_t
#define OPENSPBLAS_SPMAT_CSC spmat_csc_z_t
#define OPENSPBLAS_SPMAT_BSR spmat_bsr_z_t
#define OPENSPBLAS_SPMAT_DIA spmat_dia_z_t
#define OPENSPBLAS_SPMAT_SKY spmat_sky_z_t
#endif

#endif

/*
* values    Store the values ​​of non-zero elements in matrix A in any order, length of nnz
* row_indx  The row index of each non-zero element, the length is nnz
* col_indx  The column index of each non-zero element, the length is nnz
* rows      Number of rows of matrix
* cols      Number of column of matrix 
* nnz       The number of non-zero elements of matrix
*/

typedef struct
{
    float *values;
    OPENSPBLAS_INT *row_indx;
    OPENSPBLAS_INT *col_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
    OPENSPBLAS_INT nnz;
} spmat_coo_s_t;

typedef struct
{
    double *values;
    OPENSPBLAS_INT *row_indx;
    OPENSPBLAS_INT *col_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
    OPENSPBLAS_INT nnz;
} spmat_coo_d_t;

typedef struct
{
    OPENSPBLAS_Complex8 *values;
    OPENSPBLAS_INT *row_indx;
    OPENSPBLAS_INT *col_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
    OPENSPBLAS_INT nnz;
} spmat_coo_c_t;

typedef struct
{
    OPENSPBLAS_Complex16 *values;
    OPENSPBLAS_INT *row_indx;
    OPENSPBLAS_INT *col_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
    OPENSPBLAS_INT nnz;
} spmat_coo_z_t;

typedef struct
{
    float *values;
    OPENSPBLAS_INT *rows_start;
    OPENSPBLAS_INT *rows_end;
    OPENSPBLAS_INT *col_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
} spmat_csr_s_t;

typedef struct
{
    double *values;
    OPENSPBLAS_INT *rows_start;
    OPENSPBLAS_INT *rows_end;
    OPENSPBLAS_INT *col_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
} spmat_csr_d_t;

typedef struct
{
    OPENSPBLAS_Complex8 *values;
    OPENSPBLAS_INT *rows_start;
    OPENSPBLAS_INT *rows_end;
    OPENSPBLAS_INT *col_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
} spmat_csr_c_t;

typedef struct
{
    OPENSPBLAS_Complex16 *values;
    OPENSPBLAS_INT *rows_start;
    OPENSPBLAS_INT *rows_end;
    OPENSPBLAS_INT *col_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
} spmat_csr_z_t;

typedef struct
{
    float *values;
    OPENSPBLAS_INT *cols_start;
    OPENSPBLAS_INT *cols_end;
    OPENSPBLAS_INT *row_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
} spmat_csc_s_t;

typedef struct
{
    double *values;
    OPENSPBLAS_INT *cols_start;
    OPENSPBLAS_INT *cols_end;
    OPENSPBLAS_INT *row_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
} spmat_csc_d_t;

typedef struct
{
    OPENSPBLAS_Complex8 *values;
    OPENSPBLAS_INT *cols_start;
    OPENSPBLAS_INT *cols_end;
    OPENSPBLAS_INT *row_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
} spmat_csc_c_t;

typedef struct
{
    OPENSPBLAS_Complex16 *values;
    OPENSPBLAS_INT *cols_start;
    OPENSPBLAS_INT *cols_end;
    OPENSPBLAS_INT *row_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
} spmat_csc_z_t;

typedef struct
{
    float *values;
    OPENSPBLAS_INT *rows_start;
    OPENSPBLAS_INT *rows_end;
    OPENSPBLAS_INT *col_indx;
    OPENSPBLAS_INT rows;  // block_rows
    OPENSPBLAS_INT cols;  // block_cols
    OPENSPBLAS_INT block_size;
    openspblas_sparse_layout_t block_layout;
} spmat_bsr_s_t;

typedef struct
{
    double *values;
    OPENSPBLAS_INT *rows_start;
    OPENSPBLAS_INT *rows_end;
    OPENSPBLAS_INT *col_indx;
    OPENSPBLAS_INT rows;  // block_rows
    OPENSPBLAS_INT cols;  // block_cols
    OPENSPBLAS_INT block_size;
    openspblas_sparse_layout_t block_layout;
} spmat_bsr_d_t;

typedef struct
{
    OPENSPBLAS_Complex8 *values;
    OPENSPBLAS_INT *rows_start;
    OPENSPBLAS_INT *rows_end;
    OPENSPBLAS_INT *col_indx;
    OPENSPBLAS_INT rows;  // block_rows
    OPENSPBLAS_INT cols;  // block_cols
    OPENSPBLAS_INT block_size;
    openspblas_sparse_layout_t block_layout;
} spmat_bsr_c_t;

typedef struct
{
    OPENSPBLAS_Complex16 *values;
    OPENSPBLAS_INT *rows_start;
    OPENSPBLAS_INT *rows_end;
    OPENSPBLAS_INT *col_indx;
    OPENSPBLAS_INT rows;  // block_rows
    OPENSPBLAS_INT cols;  // block_cols
    OPENSPBLAS_INT block_size;
    openspblas_sparse_layout_t block_layout;
} spmat_bsr_z_t;

typedef struct
{
    OPENSPBLAS_INT* pointers;    
    float* values;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
    openspblas_sparse_fill_mode_t fill;
} spmat_sky_s_t;

typedef struct
{
    OPENSPBLAS_INT* pointers;    
    double* values;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
    openspblas_sparse_fill_mode_t fill;
} spmat_sky_d_t;

typedef struct
{
    OPENSPBLAS_INT* pointers;    
    OPENSPBLAS_Complex8* values;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
    openspblas_sparse_fill_mode_t fill;
} spmat_sky_c_t;

typedef struct
{
    OPENSPBLAS_INT* pointers;    
    OPENSPBLAS_Complex16* values;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
    openspblas_sparse_fill_mode_t fill;
} spmat_sky_z_t;

typedef struct
{
    float* values;
    OPENSPBLAS_INT *distance;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
    OPENSPBLAS_INT ndiag;
    OPENSPBLAS_INT lval;
}spmat_dia_s_t;


typedef struct
{
    double* values;
    OPENSPBLAS_INT *distance;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
    OPENSPBLAS_INT ndiag;
    OPENSPBLAS_INT lval;
}spmat_dia_d_t;


typedef struct
{
    OPENSPBLAS_Complex8* values;
    OPENSPBLAS_INT *distance;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
    OPENSPBLAS_INT ndiag;
    OPENSPBLAS_INT lval;
}spmat_dia_c_t;

typedef struct
{
    OPENSPBLAS_Complex16* values;
    OPENSPBLAS_INT *distance;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
    OPENSPBLAS_INT ndiag;
    OPENSPBLAS_INT lval;
}spmat_dia_z_t;
