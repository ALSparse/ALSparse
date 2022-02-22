#pragma once

/**
 *  @brief header for the internal sparse matrix definitions
 */

#include "spdef.h"
#include "types.h"

#ifndef COMPLEX

#ifndef DOUBLE
#define ALPHA_SPMAT_COO spmat_coo_s_t
#define ALPHA_SPMAT_CSR spmat_csr_s_t
#define ALPHA_SPMAT_CSC spmat_csc_s_t
#define ALPHA_SPMAT_BSR spmat_bsr_s_t
#define ALPHA_SPMAT_DIA spmat_dia_s_t
#define ALPHA_SPMAT_SKY spmat_sky_s_t
#define ALPHA_SPMAT_ELL spmat_ell_s_t
#define ALPHA_SPMAT_GEBSR spmat_gebsr_s_t
#define ALPHA_SPMAT_HYB spmat_hyb_s_t
#define ALPHA_SPMAT_CSR5 spmat_csr5_s_t
#else
#define ALPHA_SPMAT_COO spmat_coo_d_t
#define ALPHA_SPMAT_CSR spmat_csr_d_t
#define ALPHA_SPMAT_CSC spmat_csc_d_t
#define ALPHA_SPMAT_BSR spmat_bsr_d_t
#define ALPHA_SPMAT_DIA spmat_dia_d_t
#define ALPHA_SPMAT_SKY spmat_sky_d_t
#define ALPHA_SPMAT_ELL spmat_ell_d_t
#define ALPHA_SPMAT_GEBSR spmat_gebsr_d_t
#define ALPHA_SPMAT_HYB spmat_hyb_d_t
#define ALPHA_SPMAT_CSR5 spmat_csr5_d_t
#endif

#else

#ifndef DOUBLE
#define ALPHA_SPMAT_COO spmat_coo_c_t
#define ALPHA_SPMAT_CSR spmat_csr_c_t
#define ALPHA_SPMAT_CSC spmat_csc_c_t
#define ALPHA_SPMAT_BSR spmat_bsr_c_t
#define ALPHA_SPMAT_DIA spmat_dia_c_t
#define ALPHA_SPMAT_SKY spmat_sky_c_t
#define ALPHA_SPMAT_ELL spmat_ell_c_t
#define ALPHA_SPMAT_GEBSR spmat_gebsr_c_t
#define ALPHA_SPMAT_HYB spmat_hyb_c_t
#define ALPHA_SPMAT_CSR5 spmat_csr5_c_t
#else
#define ALPHA_SPMAT_COO spmat_coo_z_t
#define ALPHA_SPMAT_CSR spmat_csr_z_t
#define ALPHA_SPMAT_CSC spmat_csc_z_t
#define ALPHA_SPMAT_BSR spmat_bsr_z_t
#define ALPHA_SPMAT_DIA spmat_dia_z_t
#define ALPHA_SPMAT_SKY spmat_sky_z_t
#define ALPHA_SPMAT_ELL spmat_ell_z_t
#define ALPHA_SPMAT_GEBSR spmat_gebsr_z_t
#define ALPHA_SPMAT_HYB spmat_hyb_z_t
#define ALPHA_SPMAT_CSR5 spmat_csr5_z_t
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
  ALPHA_INT *row_indx;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT nnz;
  bool ordered;

  ALPHA_INT *d_rows_indx;
  ALPHA_INT *d_cols_indx;
  float     *d_values;
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
  ALPHA_INT *row_indx;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT nnz;
  bool ordered;

  ALPHA_INT *d_rows_indx;
  ALPHA_INT *d_cols_indx;
  double    *d_values;
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
  ALPHA_Complex8 *values;
  ALPHA_INT *row_indx;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT nnz;
  bool ordered;
  
  ALPHA_INT      *d_rows_indx;
  ALPHA_INT      *d_cols_indx;
  ALPHA_Complex8 *d_values;
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
  ALPHA_Complex16 *values;
  ALPHA_INT *row_indx;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT nnz;
  bool ordered;
  
  ALPHA_INT       *d_rows_indx;
  ALPHA_INT       *d_cols_indx;
  ALPHA_Complex16 *d_values;
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
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  bool ordered;

  float     *d_values;
  ALPHA_INT *d_row_ptr;
  ALPHA_INT *d_col_indx;

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
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  bool ordered;

  double    *d_values;
  ALPHA_INT *d_row_ptr;
  ALPHA_INT *d_col_indx;
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
  ALPHA_Complex8 *values;
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  bool ordered;

  ALPHA_Complex8 *d_values;
  ALPHA_INT      *d_row_ptr;
  ALPHA_INT      *d_col_indx;
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
  ALPHA_Complex16 *values;
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  bool ordered;

  ALPHA_Complex16 *d_values;
  ALPHA_INT       *d_row_ptr;
  ALPHA_INT       *d_col_indx;
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
  ALPHA_INT *cols_start;
  ALPHA_INT *cols_end;
  ALPHA_INT *row_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  bool ordered;
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
  ALPHA_INT *cols_start;
  ALPHA_INT *cols_end;
  ALPHA_INT *row_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  bool ordered;
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
  ALPHA_Complex8 *values;
  ALPHA_INT *cols_start;
  ALPHA_INT *cols_end;
  ALPHA_INT *row_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  bool ordered;
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
  ALPHA_Complex16 *values;
  ALPHA_INT *cols_start;
  ALPHA_INT *cols_end;
  ALPHA_INT *row_indx;
  ALPHA_INT rows;
  ALPHA_INT cols;
  bool ordered;
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
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;  // block_rows
  ALPHA_INT cols;  // block_cols
  ALPHA_INT block_size;
  alphasparse_layout_t block_layout;
  bool ordered;

  float     *d_values;
  ALPHA_INT *d_rows_ptr;
  ALPHA_INT *d_col_indx;
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
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;  // block_rows
  ALPHA_INT cols;  // block_cols
  ALPHA_INT block_size;
  alphasparse_layout_t block_layout;
  bool ordered;
  
  double    *d_values;
  ALPHA_INT *d_rows_ptr;
  ALPHA_INT *d_col_indx;
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
  ALPHA_Complex8 *values;
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;  // block_rows
  ALPHA_INT cols;  // block_cols
  ALPHA_INT block_size;
  alphasparse_layout_t block_layout;
  bool ordered;
  
  ALPHA_Complex8 *d_values;
  ALPHA_INT      *d_rows_ptr;
  ALPHA_INT      *d_col_indx;
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
  ALPHA_Complex16 *values;
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;  // block_rows
  ALPHA_INT cols;  // block_cols
  ALPHA_INT block_size;
  alphasparse_layout_t block_layout;
  bool ordered;

  
  ALPHA_Complex16 *d_values;
  ALPHA_INT       *d_rows_ptr;
  ALPHA_INT       *d_col_indx;
} spmat_bsr_z_t;

/*
* values    Data structure of the source matrix
* rows      Number of rows of matrix
* cols      Number of column of matrix 
* fill      the upper or lower triangular of the matrix
*/

typedef struct
{
  ALPHA_INT *pointers;
  float *values;
  ALPHA_INT rows;
  ALPHA_INT cols;
  alphasparse_fill_mode_t fill;
} spmat_sky_s_t;

/*
* values    Data structure of the source matrix
* rows      Number of rows of matrix
* cols      Number of column of matrix 
* fill      the upper or lower triangular of the matrix
*/

typedef struct
{
  ALPHA_INT *pointers;
  double *values;
  ALPHA_INT rows;
  ALPHA_INT cols;
  alphasparse_fill_mode_t fill;
} spmat_sky_d_t;

/*
* values    Data structure of the source matrix
* rows      Number of rows of matrix
* cols      Number of column of matrix 
* fill      the upper or lower triangular of the matrix
*/

typedef struct
{
  ALPHA_INT *pointers;
  ALPHA_Complex8 *values;
  ALPHA_INT rows;
  ALPHA_INT cols;
  alphasparse_fill_mode_t fill;
} spmat_sky_c_t;

/*
* values    Data structure of the source matrix
* rows      Number of rows of matrix
* cols      Number of column of matrix 
* fill      the upper or lower triangular of the matrix
*/

typedef struct
{
  ALPHA_INT *pointers;
  ALPHA_Complex16 *values;
  ALPHA_INT rows;
  ALPHA_INT cols;
  alphasparse_fill_mode_t fill;
} spmat_sky_z_t;

/*
* values    Data structure of the source matrix
* rows      Number of rows of matrix
* cols      Number of column of matrix 
* fill      the upper or lower triangular of the matrix
*/

typedef struct
{
  float *values;
  ALPHA_INT *distance;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT ndiag;
  ALPHA_INT lval;
} spmat_dia_s_t;

typedef struct {
  double *values;
  ALPHA_INT *distance;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT ndiag;
  ALPHA_INT lval;
} spmat_dia_d_t;

typedef struct {
  ALPHA_Complex8 *values;
  ALPHA_INT *distance;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT ndiag;
  ALPHA_INT lval;
} spmat_dia_c_t;

typedef struct {
  ALPHA_Complex16 *values;
  ALPHA_INT *distance;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT ndiag;
  ALPHA_INT lval;
} spmat_dia_z_t;

typedef struct {
  float *values;  // 列主存储非零元
  ALPHA_INT *indices;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT ld;

  float     *d_values;  // 列主存储非零元
  ALPHA_INT *d_indices;
} spmat_ell_s_t;

typedef struct {
  double *values;  // 列主存储非零元
  ALPHA_INT *indices;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT ld;

  double    *d_values;  // 列主存储非零元
  ALPHA_INT *d_indices;
} spmat_ell_d_t;

typedef struct {
  ALPHA_Complex8 *values;  // 列主存储非零元
  ALPHA_INT *indices;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT ld;

  ALPHA_Complex8 *d_values;  // 列主存储非零元
  ALPHA_INT      *d_indices;
} spmat_ell_c_t;

typedef struct {
  ALPHA_Complex16 *values;  // 列主存储非零元
  ALPHA_INT *indices;
  ALPHA_INT rows;
  ALPHA_INT cols;
  ALPHA_INT ld;

  ALPHA_Complex16 *d_values;  // 列主存储非零元
  ALPHA_INT       *d_indices;
} spmat_ell_z_t;

typedef struct {
  ALPHA_INT rows;       // number of rows (integer).
  ALPHA_INT cols;       // number of columns (integer).
  ALPHA_INT nnz;        // number of non-zero elements of the COO part (integer)
  ALPHA_INT ell_width;  // maximum number of non-zero elements per row of the ELL part (integer)
  float *ell_val;     // array of m times ell_width elements containing the ELL part data (floating
                      // point).
  ALPHA_INT *ell_col_ind;  // array of m times ell_width elements containing the ELL part column
                         // indices (integer).
  float *coo_val;        // array of nnz elements containing the COO part data (floating point).
  ALPHA_INT *coo_row_val;  // array of nnz elements containing the COO part row indices (integer).
  ALPHA_INT *coo_col_val;  // array of nnz elements containing the COO part column indices (integer).

  float     *d_ell_val;
  ALPHA_INT *d_ell_col_ind;
  float     *d_coo_val;
  ALPHA_INT *d_coo_row_val;
  ALPHA_INT *d_coo_col_val;
} spmat_hyb_s_t;

typedef struct {
  ALPHA_INT rows;       // number of rows (integer).
  ALPHA_INT cols;       // number of columns (integer).
  ALPHA_INT nnz;        // number of non-zero elements of the COO part (integer)
  ALPHA_INT ell_width;  // maximum number of non-zero elements per row of the ELL part (integer)
  double *ell_val;    // array of m times ell_width elements containing the ELL part data (floating
                      // point).
  ALPHA_INT *ell_col_ind;  // array of m times ell_width elements containing the ELL part column
                         // indices (integer).
  double *coo_val;       // array of nnz elements containing the COO part data (floating point).
  ALPHA_INT *coo_row_val;  // array of nnz elements containing the COO part row indices (integer).
  ALPHA_INT *coo_col_val;  // array of nnz elements containing the COO part column indices (integer).
  
  double    *d_ell_val;
  ALPHA_INT *d_ell_col_ind;
  double    *d_coo_val;
  ALPHA_INT *d_coo_row_val;
  ALPHA_INT *d_coo_col_val;
} spmat_hyb_d_t;

typedef struct {
  ALPHA_INT rows;           // number of rows (integer).
  ALPHA_INT cols;           // number of columns (integer).
  ALPHA_INT nnz;            // number of non-zero elements of the COO part (integer)
  ALPHA_INT ell_width;      // maximum number of non-zero elements per row of the ELL part (integer)
  ALPHA_Complex8 *ell_val;  // array of m times ell_width elements containing the ELL part data
                          // (floating point).
  ALPHA_INT *ell_col_ind;   // array of m times ell_width elements containing the ELL part column
                          // indices (integer).
  ALPHA_Complex8 *coo_val;  // array of nnz elements containing the COO part data (floating point).
  ALPHA_INT *coo_row_val;   // array of nnz elements containing the COO part row indices (integer).
  ALPHA_INT *coo_col_val;   // array of nnz elements containing the COO part column indices (integer).
  
  ALPHA_Complex8 *d_ell_val;
  ALPHA_INT      *d_ell_col_ind;
  ALPHA_Complex8 *d_coo_val;
  ALPHA_INT      *d_coo_row_val;
  ALPHA_INT      *d_coo_col_val;
} spmat_hyb_c_t;

typedef struct {
  ALPHA_INT rows;            // number of rows (integer).
  ALPHA_INT cols;            // number of columns (integer).
  ALPHA_INT nnz;             // number of non-zero elements of the COO part (integer)
  ALPHA_INT ell_width;       // maximum number of non-zero elements per row of the ELL part (integer)
  ALPHA_Complex16 *ell_val;  // array of m times ell_width elements containing the ELL part data
                           // (floating point).
  ALPHA_INT *ell_col_ind;    // array of m times ell_width elements containing the ELL part column
                           // indices (integer).
  ALPHA_Complex16 *coo_val;  // array of nnz elements containing the COO part data (floating point).
  ALPHA_INT *coo_row_val;    // array of nnz elements containing the COO part row indices (integer).
  ALPHA_INT *coo_col_val;  // array of nnz elements containing the COO part column indices (integer).
  
  ALPHA_Complex16 *d_ell_val;
  ALPHA_INT       *d_ell_col_ind;
  ALPHA_Complex16 *d_coo_val;
  ALPHA_INT       *d_coo_row_val;
  ALPHA_INT       *d_coo_col_val;
} spmat_hyb_z_t;

typedef struct {
  float *values;
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;  // block_rows
  ALPHA_INT cols;  // block_cols
  ALPHA_INT row_block_dim;
  ALPHA_INT col_block_dim;
  alphasparse_layout_t block_layout;
  bool ordered;

  float     *d_values;
  ALPHA_INT *d_rows_ptr;
  ALPHA_INT *d_col_indx;
} spmat_gebsr_s_t;

typedef struct {
  double *values;
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;  // block_rows
  ALPHA_INT cols;  // block_cols
  ALPHA_INT row_block_dim;
  ALPHA_INT col_block_dim;
  alphasparse_layout_t block_layout;
  bool ordered;

  double    *d_values;
  ALPHA_INT *d_rows_ptr;
  ALPHA_INT *d_col_indx;
} spmat_gebsr_d_t;

typedef struct {
  ALPHA_Complex8 *values;
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;  // block_rows
  ALPHA_INT cols;  // block_cols
  ALPHA_INT row_block_dim;
  ALPHA_INT col_block_dim;
  alphasparse_layout_t block_layout;
  bool ordered;
  
  ALPHA_Complex8 *d_values;
  ALPHA_INT      *d_rows_ptr;
  ALPHA_INT      *d_col_indx;
} spmat_gebsr_c_t;

typedef struct {
  ALPHA_Complex16 *values;
  ALPHA_INT *rows_start;
  ALPHA_INT *rows_end;
  ALPHA_INT *col_indx;
  ALPHA_INT rows;  // block_rows
  ALPHA_INT cols;  // block_cols
  ALPHA_INT row_block_dim;
  ALPHA_INT col_block_dim;
  alphasparse_layout_t block_layout;
  bool ordered;
    
  ALPHA_Complex16 *d_values;
  ALPHA_INT       *d_rows_ptr;
  ALPHA_INT       *d_col_indx;
} spmat_gebsr_z_t;

#define ALPHA_CSR5_OMEGA 32

typedef struct {
  ALPHA_INT num_rows;
  ALPHA_INT num_cols;
  ALPHA_INT nnz;

  ALPHA_INT csr5_sigma;               // opt: info for CSR5
  ALPHA_INT csr5_bit_y_offset;        // opt: info for CSR5
  ALPHA_INT csr5_bit_scansum_offset;  // opt: info for CSR5
  ALPHA_INT csr5_num_packets;         // opt: info for CSR5
  ALPHA_INT csr5_p;                   // opt: info for CSR5
  ALPHA_INT csr5_num_offsets;         // opt: info for CSR5
  ALPHA_INT csr5_tail_tile_start;     // opt: info for CSR5

  ALPHA_INT *col_idx;
  ALPHA_INT *row_ptr;
  float     *val;

  uint32_t  *tile_ptr;                // opt: CSR5 tile pointer CPU case
  uint32_t  *tile_desc;               // opt: CSR5 tile descriptor CPU case
  ALPHA_INT *tile_desc_offset_ptr;    // opt: CSR5 tile descriptor offset pointer CPU case
  ALPHA_INT *tile_desc_offset;        // opt: CSR5 tile descriptor offset CPU case
  float     *calibrator;              // opt: CSR5 calibrator CPU case
  
  // device
  ALPHA_INT *d_col_idx;
  ALPHA_INT *d_row_ptr;
  float     *d_val;

  uint32_t  *d_tile_ptr;
  uint32_t  *d_tile_desc;
  ALPHA_INT *d_tile_desc_offset_ptr;
  ALPHA_INT *d_tile_desc_offset;
  float     *d_calibrator;

} spmat_csr5_s_t;

typedef struct {
  ALPHA_INT num_rows;
  ALPHA_INT num_cols;
  ALPHA_INT nnz;

  ALPHA_INT *col_idx;
  ALPHA_INT *row_ptr;
  double    *val;

  uint32_t  *tile_ptr;                // opt: CSR5 tile pointer CPU case
  uint32_t  *tile_desc;               // opt: CSR5 tile descriptor CPU case
  ALPHA_INT *tile_desc_offset_ptr;    // opt: CSR5 tile descriptor offset pointer CPU case
  ALPHA_INT *tile_desc_offset;        // opt: CSR5 tile descriptor offset CPU case
  double    *calibrator;              // opt: CSR5 calibrator CPU case
  ALPHA_INT csr5_sigma;               // opt: info for CSR5
  ALPHA_INT csr5_bit_y_offset;        // opt: info for CSR5
  ALPHA_INT csr5_bit_scansum_offset;  // opt: info for CSR5
  ALPHA_INT csr5_num_packets;         // opt: info for CSR5
  ALPHA_INT csr5_p;                   // opt: info for CSR5
  ALPHA_INT csr5_num_offsets;         // opt: info for CSR5
  ALPHA_INT csr5_tail_tile_start;     // opt: info for CSR5

  // device
  ALPHA_INT *d_col_idx;
  ALPHA_INT *d_row_ptr;
  double     *d_val;

  uint32_t  *d_tile_ptr;
  uint32_t  *d_tile_desc;
  ALPHA_INT *d_tile_desc_offset_ptr;
  ALPHA_INT *d_tile_desc_offset;
  double    *d_calibrator;
} spmat_csr5_d_t;

typedef struct {
  ALPHA_INT num_rows;
  ALPHA_INT num_cols;
  ALPHA_INT nnz;

  ALPHA_INT *col_idx;
  ALPHA_INT *row_ptr;
  ALPHA_Complex8     *val;

  uint32_t  *tile_ptr;                // opt: CSR5 tile pointer CPU case
  uint32_t  *tile_desc;               // opt: CSR5 tile descriptor CPU case
  ALPHA_INT *tile_desc_offset_ptr;    // opt: CSR5 tile descriptor offset pointer CPU case
  ALPHA_INT *tile_desc_offset;        // opt: CSR5 tile descriptor offset CPU case
  ALPHA_Complex8     *calibrator;              // opt: CSR5 calibrator CPU case
  ALPHA_INT csr5_sigma;               // opt: info for CSR5
  ALPHA_INT csr5_bit_y_offset;        // opt: info for CSR5
  ALPHA_INT csr5_bit_scansum_offset;  // opt: info for CSR5
  ALPHA_INT csr5_num_packets;         // opt: info for CSR5
  ALPHA_INT csr5_p;                   // opt: info for CSR5
  ALPHA_INT csr5_num_offsets;         // opt: info for CSR5
  ALPHA_INT csr5_tail_tile_start;     // opt: info for CSR5

  // device
  ALPHA_INT *d_col_idx;
  ALPHA_INT *d_row_ptr;
  ALPHA_Complex8     *d_val;

  uint32_t       *d_tile_ptr;
  uint32_t       *d_tile_desc;
  ALPHA_INT      *d_tile_desc_offset_ptr;
  ALPHA_INT      *d_tile_desc_offset;
  ALPHA_Complex8 *d_calibrator;
} spmat_csr5_c_t;

typedef struct {
  ALPHA_INT num_rows;
  ALPHA_INT num_cols;
  ALPHA_INT nnz;

  ALPHA_INT *col_idx;
  ALPHA_INT *row_ptr;
  ALPHA_Complex16     *val;

  uint32_t  *tile_ptr;                // opt: CSR5 tile pointer CPU case
  uint32_t  *tile_desc;               // opt: CSR5 tile descriptor CPU case
  ALPHA_INT *tile_desc_offset_ptr;    // opt: CSR5 tile descriptor offset pointer CPU case
  ALPHA_INT *tile_desc_offset;        // opt: CSR5 tile descriptor offset CPU case
  ALPHA_Complex16    *calibrator;              // opt: CSR5 calibrator CPU case
  ALPHA_INT csr5_sigma;               // opt: info for CSR5
  ALPHA_INT csr5_bit_y_offset;        // opt: info for CSR5
  ALPHA_INT csr5_bit_scansum_offset;  // opt: info for CSR5
  ALPHA_INT csr5_num_packets;         // opt: info for CSR5
  ALPHA_INT csr5_p;                   // opt: info for CSR5
  ALPHA_INT csr5_num_offsets;         // opt: info for CSR5
  ALPHA_INT csr5_tail_tile_start;     // opt: info for CSR5

  // device
  ALPHA_INT *d_col_idx;
  ALPHA_INT *d_row_ptr;
  ALPHA_Complex16     *d_val;

  uint32_t        *d_tile_ptr;
  uint32_t        *d_tile_desc;
  ALPHA_INT       *d_tile_desc_offset_ptr;
  ALPHA_INT       *d_tile_desc_offset;
  ALPHA_Complex16 *d_calibrator;
} spmat_csr5_z_t;