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
    double *values;
    OPENSPBLAS_INT *row_indx;
    OPENSPBLAS_INT *col_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
    OPENSPBLAS_INT nnz;
} spmat_coo_d_t;

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
    OPENSPBLAS_Complex8 *values;
    OPENSPBLAS_INT *row_indx;
    OPENSPBLAS_INT *col_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
    OPENSPBLAS_INT nnz;
} spmat_coo_c_t;

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
    OPENSPBLAS_Complex16 *values;
    OPENSPBLAS_INT *row_indx;
    OPENSPBLAS_INT *col_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
    OPENSPBLAS_INT nnz;
} spmat_coo_z_t;

/*
* values        The value of the non-zero element in the matrix, 
                the length is equivalent to the length of col_indx
* rows_start    Contains the index of each rows of the matrix,
                rows_start[i] – ind is the starting index of the i-th row in values ​​and col_indx
* rows_end      Contains the index of each rows of the matrix, 
                rows_end[i] – ind is the end position of the i-th row in values ​​and col_indx
* col_indx      colomn index of each non-zero element of the matrix, 
                The length is at least rows_end[rows-1] – ind
* rows          Number of rows of matrix
* cols          Number of column of matrix 
*/


typedef struct
{
    float *values;
    OPENSPBLAS_INT *rows_start;
    OPENSPBLAS_INT *rows_end;
    OPENSPBLAS_INT *col_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
} spmat_csr_s_t;

/*
* values        The value of the non-zero element in the matrix, 
                the length is equivalent to the length of col_indx
* rows_start    Contains the index of each rows of the matrix,
                rows_start[i] – ind is the starting index of the i-th row in values ​​and col_indx
* rows_end      Contains the index of each rows of the matrix, 
                rows_end[i] – ind is the end position of the i-th row in values ​​and col_indx
* col_indx      colomn index of each non-zero element of the matrix, 
                The length is at least rows_end[rows-1] – ind
* rows          Number of rows of matrix
* cols          Number of column of matrix 
*/

typedef struct
{
    double *values;
    OPENSPBLAS_INT *rows_start;
    OPENSPBLAS_INT *rows_end;
    OPENSPBLAS_INT *col_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
} spmat_csr_d_t;

/*
* values        The value of the non-zero element in the matrix, 
                the length is equivalent to the length of col_indx
* rows_start    Contains the index of each rows of the matrix,
                rows_start[i] – ind is the starting index of the i-th row in values ​​and col_indx
* rows_end      Contains the index of each rows of the matrix, 
                rows_end[i] – ind is the end position of the i-th row in values ​​and col_indx
* col_indx      colomn index of each non-zero element of the matrix, 
                The length is at least rows_end[rows-1] – ind
* rows          Number of rows of matrix
* cols          Number of column of matrix 
*/


typedef struct
{
    OPENSPBLAS_Complex8 *values;
    OPENSPBLAS_INT *rows_start;
    OPENSPBLAS_INT *rows_end;
    OPENSPBLAS_INT *col_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
} spmat_csr_c_t;

/*
* values        The value of the non-zero element in the matrix, 
                the length is equivalent to the length of col_indx
* rows_start    Contains the index of each rows of the matrix,
                rows_start[i] – ind is the starting index of the i-th row in values ​​and col_indx
* rows_end      Contains the index of each rows of the matrix, 
                rows_end[i] – ind is the end position of the i-th row in values ​​and col_indx
* col_indx      colomn index of each non-zero element of the matrix, 
                The length is at least rows_end[rows-1] – ind
* rows          Number of rows of matrix
* cols          Number of column of matrix 
*/

typedef struct
{
    OPENSPBLAS_Complex16 *values;
    OPENSPBLAS_INT *rows_start;
    OPENSPBLAS_INT *rows_end;
    OPENSPBLAS_INT *col_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
} spmat_csr_z_t;

/*
* values        The value of the non-zero element in the matrix, 
                the length is equivalent to the length of row_indx
* cols_start    Contains the index of each column of the matrix,
                rows_start[i] – ind is the starting index of the i-th column in values ​​and col_indx
* cols_end      Contains the index of each column of the matrix, 
                rows_end[i] – ind is the end position of the i-th column in values ​​and col_indx
* row_indx      row index of each non-zero element of the matrix,
                the length is at least cols_end[cols-1] – ind
* rows          Number of rows of matrix
* cols          Number of column of matrix 
*/

typedef struct
{
    float *values;
    OPENSPBLAS_INT *cols_start;
    OPENSPBLAS_INT *cols_end;
    OPENSPBLAS_INT *row_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
} spmat_csc_s_t;
/*
* values        The value of the non-zero element in the matrix, 
                the length is equivalent to the length of row_indx
* cols_start    Contains the index of each column of the matrix,
                rows_start[i] – ind is the starting index of the i-th column in values ​​and col_indx
* cols_end      Contains the index of each column of the matrix, 
                rows_end[i] – ind is the end position of the i-th column in values ​​and col_indx
* row_indx      row index of each non-zero element of the matrix,
                the length is at least cols_end[cols-1] – ind
* rows          Number of rows of matrix
* cols          Number of column of matrix 
*/
typedef struct
{
    double *values;
    OPENSPBLAS_INT *cols_start;
    OPENSPBLAS_INT *cols_end;
    OPENSPBLAS_INT *row_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
} spmat_csc_d_t;
/*
* values        The value of the non-zero element in the matrix, 
                the length is equivalent to the length of row_indx
* cols_start    Contains the index of each column of the matrix,
                rows_start[i] – ind is the starting index of the i-th column in values ​​and col_indx
* cols_end      Contains the index of each column of the matrix, 
                rows_end[i] – ind is the end position of the i-th column in values ​​and col_indx
* row_indx      row index of each non-zero element of the matrix,
                the length is at least cols_end[cols-1] – ind
* rows          Number of rows of matrix
* cols          Number of column of matrix 
*/
typedef struct
{
    OPENSPBLAS_Complex8 *values;
    OPENSPBLAS_INT *cols_start;
    OPENSPBLAS_INT *cols_end;
    OPENSPBLAS_INT *row_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
} spmat_csc_c_t;
/*
* values        The value of the non-zero element in the matrix, 
                the length is equivalent to the length of row_indx
* cols_start    Contains the index of each column of the matrix,
                rows_start[i] – ind is the starting index of the i-th column in values ​​and col_indx
* cols_end      Contains the index of each column of the matrix, 
                rows_end[i] – ind is the end position of the i-th column in values ​​and col_indx
* row_indx      row index of each non-zero element of the matrix,
                the length is at least cols_end[cols-1] – ind
* rows          Number of rows of matrix
* cols          Number of column of matrix 
*/
typedef struct
{
    OPENSPBLAS_Complex16 *values;
    OPENSPBLAS_INT *cols_start;
    OPENSPBLAS_INT *cols_end;
    OPENSPBLAS_INT *row_indx;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
} spmat_csc_z_t;

/*
* values        The value of the non-zero element in the matrix, 
                the length is equivalent to the length of col_indx multiplied by block_size*block_size
* rows_start    index of each non-zero block row of the matrix 
                rows_start[i] – ind is the starting index of the i-th block row in values ​​and col_indx
* rows_end      index of each non-zero block row of the matrix
                rows_end[i] – ind is the end position of the i-th block row in values ​​and col_indx
* col_indx      row index of each non-zero element block of the matrix; 
                the length is at least rows_end[rows-1] – ind
* rows          Number of rows of non-zero block of matrix
* cols          Number of colomns of non-zero block of matrix
* block_size    length of the non-zero element block of the matrix, 
                the size of each non-zero element block is block_size * block_size
* block_layout  Describe the storage mode of non-zero elements in a sparse matrix block:
                row_major or colomn_major
*
*/

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

/*
* values        The value of the non-zero element in the matrix, 
                the length is equivalent to the length of col_indx multiplied by block_size*block_size
* rows_start    index of each non-zero block row of the matrix 
                rows_start[i] – ind is the starting index of the i-th block row in values ​​and col_indx
* rows_end      index of each non-zero block row of the matrix
                rows_end[i] – ind is the end position of the i-th block row in values ​​and col_indx
* col_indx      row index of each non-zero element block of the matrix; 
                the length is at least rows_end[rows-1] – ind
* rows          Number of rows of non-zero block of matrix
* cols          Number of colomns of non-zero block of matrix
* block_size    length of the non-zero element block of the matrix, 
                the size of each non-zero element block is block_size * block_size
* block_layout  Describe the storage mode of non-zero elements in a sparse matrix block:
                row_major or colomn_major
*
*/

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

/*
* values        The value of the non-zero element in the matrix, 
                the length is equivalent to the length of col_indx multiplied by block_size*block_size
* rows_start    index of each non-zero block row of the matrix 
                rows_start[i] – ind is the starting index of the i-th block row in values ​​and col_indx
* rows_end      index of each non-zero block row of the matrix
                rows_end[i] – ind is the end position of the i-th block row in values ​​and col_indx
* col_indx      row index of each non-zero element block of the matrix; 
                the length is at least rows_end[rows-1] – ind
* rows          Number of rows of non-zero block of matrix
* cols          Number of colomns of non-zero block of matrix
* block_size    length of the non-zero element block of the matrix, 
                the size of each non-zero element block is block_size * block_size
* block_layout  Describe the storage mode of non-zero elements in a sparse matrix block:
                row_major or colomn_major
*
*/

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

/*
* values        The value of the non-zero element in the matrix, 
                the length is equivalent to the length of col_indx multiplied by block_size*block_size
* rows_start    index of each non-zero block row of the matrix 
                rows_start[i] – ind is the starting index of the i-th block row in values ​​and col_indx
* rows_end      index of each non-zero block row of the matrix
                rows_end[i] – ind is the end position of the i-th block row in values ​​and col_indx
* col_indx      row index of each non-zero element block of the matrix; 
                the length is at least rows_end[rows-1] – ind
* rows          Number of rows of non-zero block of matrix
* cols          Number of colomns of non-zero block of matrix
* block_size    length of the non-zero element block of the matrix, 
                the size of each non-zero element block is block_size * block_size
* block_layout  Describe the storage mode of non-zero elements in a sparse matrix block:
                row_major or colomn_major
*
*/

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

/*
* values    Data structure of the source matrix
* rows      Number of rows of matrix
* cols      Number of column of matrix 
* fill      the upper or lower triangular of the matrix
*/

typedef struct
{
    OPENSPBLAS_INT* pointers;    
    float* values;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
    openspblas_sparse_fill_mode_t fill;
} spmat_sky_s_t;

/*
* values    Data structure of the source matrix
* rows      Number of rows of matrix
* cols      Number of column of matrix 
* fill      the upper or lower triangular of the matrix
*/

typedef struct
{
    OPENSPBLAS_INT* pointers;    
    double* values;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
    openspblas_sparse_fill_mode_t fill;
} spmat_sky_d_t;

/*
* values    Data structure of the source matrix
* rows      Number of rows of matrix
* cols      Number of column of matrix 
* fill      the upper or lower triangular of the matrix
*/

typedef struct
{
    OPENSPBLAS_INT* pointers;    
    OPENSPBLAS_Complex8* values;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
    openspblas_sparse_fill_mode_t fill;
} spmat_sky_c_t;

/*
* values    Data structure of the source matrix
* rows      Number of rows of matrix
* cols      Number of column of matrix 
* fill      the upper or lower triangular of the matrix
*/

typedef struct
{
    OPENSPBLAS_INT* pointers;    
    OPENSPBLAS_Complex16* values;
    OPENSPBLAS_INT rows;
    OPENSPBLAS_INT cols;
    openspblas_sparse_fill_mode_t fill;
} spmat_sky_z_t;

/*
* values    Data structure of the source matrix
* rows      Number of rows of matrix
* cols      Number of column of matrix 
* fill      the upper or lower triangular of the matrix
*/

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
