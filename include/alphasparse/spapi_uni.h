
#pragma once

/**
 * @brief header for all spblas user interfaces
 */

#include "spdef.h"
#include "types.h"

/**
 * ----------------------------------------------------------------------------
 */

// alphasparse_status_t alphasparse_transpose(const alphasparse_matrix_t source, alphasparse_matrix_t *dest);

// alphasparse_status_t alphasparse_convert_csc(const alphasparse_matrix_t source,
//                                            const alphasparse_operation_t operation,
//                                            alphasparse_matrix_t *dest);

// alphasparse_status_t alphasparse_convert_bsr(const alphasparse_matrix_t source, /* convert original matrix to BSR representation */
//                                            const ALPHA_INT block_size,
//                                            const alphasparse_layout_t block_layout, /* block storage: row-major or column-major */
//                                            const alphasparse_operation_t operation, /* as is, transposed or conjugate transposed */
//                                            alphasparse_matrix_t *dest);

// alphasparse_status_t alphasparse_convert_sky(const alphasparse_matrix_t source,
//                                            const alphasparse_operation_t operation,
//                                            const alphasparse_fill_mode_t fill,
//                                            alphasparse_matrix_t *dest);

// alphasparse_status_t alphasparse_convert_dia(const alphasparse_matrix_t source,
//                                            const alphasparse_operation_t operation,
//                                            alphasparse_matrix_t *dest);

// alphasparse_status_t alphasparse_convert_ell(const alphasparse_matrix_t source,
//                                            const alphasparse_operation_t operation,
//                                            alphasparse_matrix_t *dest);

// alphasparse_status_t alphasparse_convert_hyb(const alphasparse_matrix_t source,
//                                            const alphasparse_operation_t operation,
//                                            alphasparse_matrix_t *dest);

// alphasparse_status_t alphasparse_convert_csr5(const alphasparse_matrix_t source,
//                                            const alphasparse_operation_t operation,
//                                            alphasparse_matrix_t *dest);

// alphasparse_status_t alphasparse_convert_hints_bsr(const alphasparse_matrix_t source, /* convert original matrix to BSR representation */
//                                                  const ALPHA_INT block_size,
//                                                  const alphasparse_layout_t block_layout, /* block storage: row-major or column-major */
//                                                  const alphasparse_operation_t operation, /* as is, transposed or conjugate transposed */
//                                                  alphasparse_matrix_t *dest);

// alphasparse_status_t alphasparse_convert_hints_dia(const alphasparse_matrix_t source,
//                                                  const alphasparse_operation_t operation,
//                                                  alphasparse_matrix_t *dest);

// alphasparse_status_t alphasparse_convert_hints_ell(const alphasparse_matrix_t source,
//                                                  const alphasparse_operation_t operation,
//                                                  alphasparse_matrix_t *dest);

// alphasparse_status_t alphasparse_convert_gebsr(const alphasparse_matrix_t source, /* convert original matrix to GEBSR representation */
//                                              const ALPHA_INT block_row_dim,
//                                              const ALPHA_INT block_col_dim,
//                                              const alphasparse_layout_t block_layout, /* block storage: row-major or column-major */
//                                              const alphasparse_operation_t operation, /* as is, transposed or conjugate transposed */
//                                              alphasparse_matrix_t *dest);

/**
 * --------------------------------------------------------------------------------------
 */

/*****************************************************************************************/
/*************************************** Creation routines *******************************/
/*****************************************************************************************/

/*
    Matrix handle is used for storing information about the matrix and matrix values

    Create matrix from one of the existing sparse formats by creating the handle with matrix info and copy matrix values if requested.
    Collect high-level info about the matrix. Need to use this interface for the case with several calls in program for performance reasons,
    where optimizations are not required.

    coordinate format,
    ALPHA_SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are stored in the handle

    *** User data is not marked const since the alphasparse_order() or alphasparse_?_set_values()
    functionality could change user data.  However, this is only done by a user call. 
    Internally const-ness of user data is maintained other than through explicit
    use of these interfaces.

*/
alphasparse_status_t alphasparse_uni_s_create_coo(alphasparse_matrix_t *A,
                                            const alphasparse_executor_t exe,
                                            const alphasparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            const ALPHA_INT nnz,
                                            ALPHA_INT *row_indx,
                                            ALPHA_INT *col_indx,
                                            float *values);

alphasparse_status_t alphasparse_uni_d_create_coo(alphasparse_matrix_t *A,
                                            const alphasparse_executor_t exe,
                                            const alphasparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            const ALPHA_INT nnz,
                                            ALPHA_INT *row_indx,
                                            ALPHA_INT *col_indx,
                                            double *values);

alphasparse_status_t alphasparse_uni_c_create_coo(alphasparse_matrix_t *A,
                                            const alphasparse_executor_t exe,
                                            const alphasparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            const ALPHA_INT nnz,
                                            ALPHA_INT *row_indx,
                                            ALPHA_INT *col_indx,
                                            ALPHA_Complex8 *values);

alphasparse_status_t alphasparse_uni_z_create_coo(alphasparse_matrix_t *A,
                                            const alphasparse_executor_t exe,
                                            const alphasparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            const ALPHA_INT nnz,
                                            ALPHA_INT *row_indx,
                                            ALPHA_INT *col_indx,
                                            ALPHA_Complex16 *values);

/*
    compressed sparse row format (4-arrays version),
    ALPHA_SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are stored in the handle

    *** User data is not marked const since the alphasparse_order() or alphasparse_?_set_values()
    functionality could change user data.  However, this is only done by a user call. 
    Internally const-ness of user data is maintained other than through explicit
    use of these interfaces.


*/
// alphasparse_status_t alphasparse_s_create_csr(alphasparse_matrix_t *A,
//                                             const alphasparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
//                                             const ALPHA_INT rows,
//                                             const ALPHA_INT cols,
//                                             ALPHA_INT *rows_start,
//                                             ALPHA_INT *rows_end,
//                                             ALPHA_INT *col_indx,
//                                             float *values);

// alphasparse_status_t alphasparse_d_create_csr(alphasparse_matrix_t *A,
//                                             const alphasparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
//                                             const ALPHA_INT rows,
//                                             const ALPHA_INT cols,
//                                             ALPHA_INT *rows_start,
//                                             ALPHA_INT *rows_end,
//                                             ALPHA_INT *col_indx,
//                                             double *values);

// alphasparse_status_t alphasparse_c_create_csr(alphasparse_matrix_t *A,
//                                             const alphasparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
//                                             const ALPHA_INT rows,
//                                             const ALPHA_INT cols,
//                                             ALPHA_INT *rows_start,
//                                             ALPHA_INT *rows_end,
//                                             ALPHA_INT *col_indx,
//                                             ALPHA_Complex8 *values);

// alphasparse_status_t alphasparse_z_create_csr(alphasparse_matrix_t *A,
//                                             const alphasparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
//                                             const ALPHA_INT rows,
//                                             const ALPHA_INT cols,
//                                             ALPHA_INT *rows_start,
//                                             ALPHA_INT *rows_end,
//                                             ALPHA_INT *col_indx,
//                                             ALPHA_Complex16 *values);

// /*
//     compressed sparse column format (4-arrays version),
//     ALPHA_SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are stored in the handle

//     *** User data is not marked const since the alphasparse_order() or alphasparse_?_set_values()
//     functionality could change user data.  However, this is only done by a user call. 
//     Internally const-ness of user data is maintained other than through explicit
//     use of these interfaces.

// */
// alphasparse_status_t alphasparse_s_create_csc(alphasparse_matrix_t *A,
//                                             const alphasparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
//                                             const ALPHA_INT rows,
//                                             const ALPHA_INT cols,
//                                             ALPHA_INT *cols_start,
//                                             ALPHA_INT *cols_end,
//                                             ALPHA_INT *row_indx,
//                                             float *values);

// alphasparse_status_t alphasparse_d_create_csc(alphasparse_matrix_t *A,
//                                             const alphasparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
//                                             const ALPHA_INT rows,
//                                             const ALPHA_INT cols,
//                                             ALPHA_INT *cols_start,
//                                             ALPHA_INT *cols_end,
//                                             ALPHA_INT *row_indx,
//                                             double *values);

// alphasparse_status_t alphasparse_c_create_csc(alphasparse_matrix_t *A,
//                                             const alphasparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
//                                             const ALPHA_INT rows,
//                                             const ALPHA_INT cols,
//                                             ALPHA_INT *cols_start,
//                                             ALPHA_INT *cols_end,
//                                             ALPHA_INT *row_indx,
//                                             ALPHA_Complex8 *values);

// alphasparse_status_t alphasparse_z_create_csc(alphasparse_matrix_t *A,
//                                             const alphasparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
//                                             const ALPHA_INT rows,
//                                             const ALPHA_INT cols,
//                                             ALPHA_INT *cols_start,
//                                             ALPHA_INT *cols_end,
//                                             ALPHA_INT *row_indx,
//                                             ALPHA_Complex16 *values);

// /*
//     compressed block sparse row format (4-arrays version, square blocks),
//     ALPHA_SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are stored in the handle

//     *** User data is not marked const since the alphasparse_order() or alphasparse_?_set_values()
//     functionality could change user data.  However, this is only done by a user call. 
//     Internally const-ness of user data is maintained other than through explicit
//     use of these interfaces.

// */
// alphasparse_status_t alphasparse_s_create_bsr(alphasparse_matrix_t *A,
//                                             const alphasparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
//                                             const alphasparse_layout_t block_layout, /* block storage: row-major or column-major */
//                                             const ALPHA_INT rows,
//                                             const ALPHA_INT cols,
//                                             const ALPHA_INT block_size,
//                                             ALPHA_INT *rows_start,
//                                             ALPHA_INT *rows_end,
//                                             ALPHA_INT *col_indx,
//                                             float *values);

// alphasparse_status_t alphasparse_d_create_bsr(alphasparse_matrix_t *A,
//                                             const alphasparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
//                                             const alphasparse_layout_t block_layout, /* block storage: row-major or column-major */
//                                             const ALPHA_INT rows,
//                                             const ALPHA_INT cols,
//                                             const ALPHA_INT block_size,
//                                             ALPHA_INT *rows_start,
//                                             ALPHA_INT *rows_end,
//                                             ALPHA_INT *col_indx,
//                                             double *values);

// alphasparse_status_t alphasparse_c_create_bsr(alphasparse_matrix_t *A,
//                                             const alphasparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
//                                             const alphasparse_layout_t block_layout, /* block storage: row-major or column-major */
//                                             const ALPHA_INT rows,
//                                             const ALPHA_INT cols,
//                                             const ALPHA_INT block_size,
//                                             ALPHA_INT *rows_start,
//                                             ALPHA_INT *rows_end,
//                                             ALPHA_INT *col_indx,
//                                             ALPHA_Complex8 *values);

// alphasparse_status_t alphasparse_z_create_bsr(alphasparse_matrix_t *A,
//                                             const alphasparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
//                                             const alphasparse_layout_t block_layout, /* block storage: row-major or column-major */
//                                             const ALPHA_INT rows,
//                                             const ALPHA_INT cols,
//                                             const ALPHA_INT block_size,
//                                             ALPHA_INT *rows_start,
//                                             ALPHA_INT *rows_end,
//                                             ALPHA_INT *col_indx,
//                                             ALPHA_Complex16 *values);

// /*
//     Create copy of the existing handle; matrix properties could be changed.
//     For example it could be used for extracting triangular or diagonal parts from existing matrix.
// */
// alphasparse_status_t alphasparse_copy(const alphasparse_matrix_t source,
//                                     const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                     alphasparse_matrix_t *dest);

// /*
//     destroy matrix handle; if sparse matrix was stored inside the handle it also deallocates the matrix
//     It is user's responsibility not to delete the handle with the matrix, if this matrix is shared with other handles
// */
// alphasparse_status_t alphasparse_destroy(alphasparse_matrix_t A);
// /*
//     return extended error information from last operation;
//     e.g. info about wrong input parameter, memory sizes that couldn't be allocated
// */
// alphasparse_status_t alphasparse_get_error_info(alphasparse_matrix_t A, ALPHA_INT *info); /* unsupported currently */

/*****************************************************************************************/
/************************ Converters of internal representation  *************************/
/*****************************************************************************************/

/* converters from current format to another */
alphasparse_status_t alphasparse_uni_convert_csr(const alphasparse_matrix_t source,       /* convert original matrix to CSR representation */
                                           const alphasparse_operation_t operation, /* as is, transposed or conjugate transposed */
                                           alphasparse_matrix_t *dest);

alphasparse_status_t alphasparse_uni_convert_coo(const alphasparse_matrix_t source,       /* convert original matrix to CSR representation */
                                           const alphasparse_operation_t operation, /* as is, transposed or conjugate transposed */
                                           alphasparse_matrix_t *dest);

// alphasparse_status_t alphasparse_s_export_bsr(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             alphasparse_layout_t *block_layout, /* block storage: row-major or column-major */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT *block_size,
//                                             ALPHA_INT **rows_start,
//                                             ALPHA_INT **rows_end,
//                                             ALPHA_INT **col_indx,
//                                             float **values);

// alphasparse_status_t alphasparse_d_export_bsr(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             alphasparse_layout_t *block_layout, /* block storage: row-major or column-major */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT *block_size,
//                                             ALPHA_INT **rows_start,
//                                             ALPHA_INT **rows_end,
//                                             ALPHA_INT **col_indx,
//                                             double **values);

// alphasparse_status_t alphasparse_c_export_bsr(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             alphasparse_layout_t *block_layout, /* block storage: row-major or column-major */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT *block_size,
//                                             ALPHA_INT **rows_start,
//                                             ALPHA_INT **rows_end,
//                                             ALPHA_INT **col_indx,
//                                             ALPHA_Complex8 **values);

// alphasparse_status_t alphasparse_z_export_bsr(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             alphasparse_layout_t *block_layout, /* block storage: row-major or column-major */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT *block_size,
//                                             ALPHA_INT **rows_start,
//                                             ALPHA_INT **rows_end,
//                                             ALPHA_INT **col_indx,
//                                             ALPHA_Complex16 **values);

// alphasparse_status_t alphasparse_s_export_csr(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT **rows_start,
//                                             ALPHA_INT **rows_end,
//                                             ALPHA_INT **col_indx,
//                                             float **values);

// alphasparse_status_t alphasparse_d_export_csr(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT **rows_start,
//                                             ALPHA_INT **rows_end,
//                                             ALPHA_INT **col_indx,
//                                             double **values);

// alphasparse_status_t alphasparse_c_export_csr(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT **rows_start,
//                                             ALPHA_INT **rows_end,
//                                             ALPHA_INT **col_indx,
//                                             ALPHA_Complex8 **values);

// alphasparse_status_t alphasparse_z_export_csr(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT **rows_start,
//                                             ALPHA_INT **rows_end,
//                                             ALPHA_INT **col_indx,
//                                             ALPHA_Complex16 **values);

// alphasparse_status_t alphasparse_s_export_csc(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT **cols_start,
//                                             ALPHA_INT **cols_end,
//                                             ALPHA_INT **row_indx,
//                                             float **values);

// alphasparse_status_t alphasparse_d_export_csc(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT **cols_start,
//                                             ALPHA_INT **cols_end,
//                                             ALPHA_INT **row_indx,
//                                             double **values);

// alphasparse_status_t alphasparse_c_export_csc(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT **cols_start,
//                                             ALPHA_INT **cols_end,
//                                             ALPHA_INT **row_indx,
//                                             ALPHA_Complex8 **values);

// alphasparse_status_t alphasparse_z_export_csc(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT **cols_start,
//                                             ALPHA_INT **cols_end,
//                                             ALPHA_INT **row_indx,
//                                             ALPHA_Complex16 **values);

// alphasparse_status_t alphasparse_s_export_ell(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT *width,
//                                             ALPHA_INT **col_indx,
//                                             float **values);

// alphasparse_status_t alphasparse_d_export_ell(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT *width,
//                                             ALPHA_INT **col_indx,
//                                             double **values);

// alphasparse_status_t alphasparse_c_export_ell(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT *width,
//                                             ALPHA_INT **col_indx,
//                                             ALPHA_Complex8 **values);

// alphasparse_status_t alphasparse_z_export_ell(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT *width,
//                                             ALPHA_INT **col_indx,
//                                             ALPHA_Complex16 **values);

// alphasparse_status_t alphasparse_s_export_gebsr(const alphasparse_matrix_t source,
//                                               alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                               alphasparse_layout_t *block_layout, /* block storage: row-major or column-major */
//                                               ALPHA_INT *rows,
//                                               ALPHA_INT *cols,
//                                               ALPHA_INT *block_row_dim,
//                                               ALPHA_INT *block_col_dim,
//                                               ALPHA_INT **rows_start,
//                                               ALPHA_INT **rows_end,
//                                               ALPHA_INT **col_indx,
//                                               float **values);

// alphasparse_status_t alphasparse_d_export_gebsr(const alphasparse_matrix_t source,
//                                               alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                               alphasparse_layout_t *block_layout, /* block storage: row-major or column-major */
//                                               ALPHA_INT *rows,
//                                               ALPHA_INT *cols,
//                                               ALPHA_INT *block_row_dim,
//                                               ALPHA_INT *block_col_dim,
//                                               ALPHA_INT **rows_start,
//                                               ALPHA_INT **rows_end,
//                                               ALPHA_INT **col_indx,
//                                               double **values);

// alphasparse_status_t alphasparse_c_export_gebsr(const alphasparse_matrix_t source,
//                                               alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                               alphasparse_layout_t *block_layout, /* block storage: row-major or column-major */
//                                               ALPHA_INT *rows,
//                                               ALPHA_INT *cols,
//                                               ALPHA_INT *block_row_dim,
//                                               ALPHA_INT *block_col_dim,
//                                               ALPHA_INT **rows_start,
//                                               ALPHA_INT **rows_end,
//                                               ALPHA_INT **col_indx,
//                                               ALPHA_Complex8 **values);

// alphasparse_status_t alphasparse_z_export_gebsr(const alphasparse_matrix_t source,
//                                               alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                               alphasparse_layout_t *block_layout, /* block storage: row-major or column-major */
//                                               ALPHA_INT *rows,
//                                               ALPHA_INT *cols,
//                                               ALPHA_INT *block_row_dim,
//                                               ALPHA_INT *block_col_dim,
//                                               ALPHA_INT **rows_start,
//                                               ALPHA_INT **rows_end,
//                                               ALPHA_INT **col_indx,
//                                               ALPHA_Complex16 **values);

// alphasparse_status_t alphasparse_s_export_hyb(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT *nnz,
//                                             ALPHA_INT *ell_width,
//                                             float **ell_val,
//                                             ALPHA_INT **ell_col_ind,
//                                             float **coo_val,
//                                             ALPHA_INT **coo_row_val,
//                                             ALPHA_INT **coo_col_val);

// alphasparse_status_t alphasparse_d_export_hyb(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT *nnz,
//                                             ALPHA_INT *ell_width,
//                                             double **ell_val,
//                                             ALPHA_INT **ell_col_ind,
//                                             double **coo_val,
//                                             ALPHA_INT **coo_row_val,
//                                             ALPHA_INT **coo_col_val);

// alphasparse_status_t alphasparse_c_export_hyb(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT *nnz,
//                                             ALPHA_INT *ell_width,
//                                             ALPHA_Complex8 **ell_val,
//                                             ALPHA_INT **ell_col_ind,
//                                             ALPHA_Complex8 **coo_val,
//                                             ALPHA_INT **coo_row_val,
//                                             ALPHA_INT **coo_col_val);

// alphasparse_status_t alphasparse_z_export_hyb(const alphasparse_matrix_t source,
//                                             alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
//                                             ALPHA_INT *rows,
//                                             ALPHA_INT *cols,
//                                             ALPHA_INT *nnz,
//                                             ALPHA_INT *ell_width,
//                                             ALPHA_Complex16 **ell_val,
//                                             ALPHA_INT **ell_col_ind,
//                                             ALPHA_Complex16 **coo_val,
//                                             ALPHA_INT **coo_row_val,
//                                             ALPHA_INT **coo_col_val);

// /*****************************************************************************************/
// /************************** Step-by-step modification routines ***************************/
// /*****************************************************************************************/

// /* update existing value in the matrix ( for internal storage only, should not work with user-allocated matrices) */
// alphasparse_status_t alphasparse_s_set_value(alphasparse_matrix_t A,
//                                            const ALPHA_INT row,
//                                            const ALPHA_INT col,
//                                            const float value);

// alphasparse_status_t alphasparse_d_set_value(alphasparse_matrix_t A,
//                                            const ALPHA_INT row,
//                                            const ALPHA_INT col,
//                                            const double value);

// alphasparse_status_t alphasparse_c_set_value(alphasparse_matrix_t A,
//                                            const ALPHA_INT row,
//                                            const ALPHA_INT col,
//                                            const ALPHA_Complex8 value);

// alphasparse_status_t alphasparse_z_set_value(alphasparse_matrix_t A,
//                                            const ALPHA_INT row,
//                                            const ALPHA_INT col,
//                                            const ALPHA_Complex16 value);

// /* update existing values in the matrix for internal storage only 
//        can be used to either update all or selected values */
// alphasparse_status_t alphasparse_s_update_values(alphasparse_matrix_t A,
//                                                const ALPHA_INT nvalues,
//                                                const ALPHA_INT *indx,
//                                                const ALPHA_INT *indy,
//                                                float *values);

// alphasparse_status_t alphasparse_d_update_values(alphasparse_matrix_t A,
//                                                const ALPHA_INT nvalues,
//                                                const ALPHA_INT *indx,
//                                                const ALPHA_INT *indy,
//                                                double *values);

// alphasparse_status_t alphasparse_c_update_values(alphasparse_matrix_t A,
//                                                const ALPHA_INT nvalues,
//                                                const ALPHA_INT *indx,
//                                                const ALPHA_INT *indy,
//                                                ALPHA_Complex8 *values);

// alphasparse_status_t alphasparse_z_update_values(alphasparse_matrix_t A,
//                                                const ALPHA_INT nvalues,
//                                                const ALPHA_INT *indx,
//                                                const ALPHA_INT *indy,
//                                                ALPHA_Complex16 *values);

// /*****************************************************************************************/
// /****************************** Verbose mode routine *************************************/
// /*****************************************************************************************/

// /* allow to switch on/off verbose mode */
// alphasparse_status_t alphasparse_set_verbose_mode(alpha_verbose_mode_t verbose); /* unsupported currently */

// /*****************************************************************************************/
// /****************************** Optimization routines ************************************/
// /*****************************************************************************************/

// /* Describe expected operations with amount of iterations */
// alphasparse_status_t alphasparse_set_mv_hint(const alphasparse_matrix_t A,
//                                            const alphasparse_operation_t operation, /* ALPHA_SPARSE_OPERATION_NON_TRANSPOSE is default value for infinite amount of calls */
//                                            const struct alpha_matrix_descr descr,    /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                            const ALPHA_INT expected_calls);

// alphasparse_status_t alphasparse_set_dotmv_hint(const alphasparse_matrix_t A,
//                                               const alphasparse_operation_t operation, /* ALPHA_SPARSE_OPERATION_NON_TRANSPOSE is default value for infinite amount of calls */
//                                               const struct alpha_matrix_descr descr,    /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                               const ALPHA_INT expectedCalls);

// alphasparse_status_t alphasparse_set_mmd_hint(const alphasparse_matrix_t A,
//                                             const alphasparse_operation_t operation,
//                                             const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                             const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
//                                             const ALPHA_INT dense_matrix_size,     /* amount of columns in dense matrix */
//                                             const ALPHA_INT expected_calls);

// alphasparse_status_t alphasparse_set_sv_hint(const alphasparse_matrix_t A,
//                                            const alphasparse_operation_t operation, /* ALPHA_SPARSE_OPERATION_NON_TRANSPOSE is default value for infinite amount of calls */
//                                            const struct alpha_matrix_descr descr,    /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                            const ALPHA_INT expected_calls);

// alphasparse_status_t alphasparse_set_sm_hint(const alphasparse_matrix_t A,
//                                            const alphasparse_operation_t operation,
//                                            const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                            const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
//                                            const ALPHA_INT dense_matrix_size,     /* amount of columns in dense matrix */
//                                            const ALPHA_INT expected_calls);

// alphasparse_status_t alphasparse_set_mm_hint(const alphasparse_matrix_t A,
//                                            const alphasparse_operation_t transA,
//                                            const struct alpha_matrix_descr descrA, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                            const alphasparse_matrix_t B,
//                                            const alphasparse_operation_t transB,
//                                            const struct alpha_matrix_descr descrB,
//                                            const ALPHA_INT expected_calls);

// alphasparse_status_t alphasparse_set_symgs_hint(const alphasparse_matrix_t A,
//                                               const alphasparse_operation_t operation, /* ALPHA_SPARSE_OPERATION_NON_TRANSPOSE is default value for infinite amount of calls */
//                                               const struct alpha_matrix_descr descr,    /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                               const ALPHA_INT expected_calls);

// alphasparse_status_t alphasparse_set_lu_smoother_hint(const alphasparse_matrix_t A,
//                                                     const alphasparse_operation_t operation,
//                                                     const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                                     const ALPHA_INT expectedCalls);

// /* Describe memory usage model */
// alphasparse_status_t alphasparse_set_memory_hint(const alphasparse_matrix_t A,
//                                                const alphasparse_memory_usage_t policy); /* ALPHA_SPARSE_MEMORY_AGGRESSIVE is default value */

// /*
//     Optimize matrix described by the handle. It uses hints (optimization and memory) that should be set up before this call.
//     If hints were not explicitly defined, default vales are:
//     ALPHA_SPARSE_OPERATION_NON_TRANSPOSE for matrix-vector multiply with infinite number of expected iterations.
// */
// alphasparse_status_t alphasparse_optimize(alphasparse_matrix_t A);

// /*****************************************************************************************/
// /****************************** Computational routines ***********************************/
// /*****************************************************************************************/

// alphasparse_status_t alphasparse_order(const alphasparse_matrix_t A);

/*
    Perform computations based on created matrix handle

    Level 2
*/
/*   Computes y = alpha * A * x + beta * y   */
alphasparse_status_t alphasparse_uni_s_mv(const alphasparse_operation_t operation,
                                    const float alpha,
                                    const alphasparse_matrix_t A,
                                    const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                    const float *x,
                                    const float beta,
                                    float *y);

alphasparse_status_t alphasparse_uni_d_mv(const alphasparse_operation_t operation,
                                    const double alpha,
                                    const alphasparse_matrix_t A,
                                    const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                    const double *x,
                                    const double beta,
                                    double *y);

alphasparse_status_t alphasparse_uni_c_mv(const alphasparse_operation_t operation,
                                    const ALPHA_Complex8 alpha,
                                    const alphasparse_matrix_t A,
                                    const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                    const ALPHA_Complex8 *x,
                                    const ALPHA_Complex8 beta,
                                    ALPHA_Complex8 *y);

alphasparse_status_t alphasparse_uni_z_mv(const alphasparse_operation_t operation,
                                    const ALPHA_Complex16 alpha,
                                    const alphasparse_matrix_t A,
                                    const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                    const ALPHA_Complex16 *x,
                                    const ALPHA_Complex16 beta,
                                    ALPHA_Complex16 *y);

// /*    Computes y = alpha * A * x + beta * y  and d = <x, y> , the l2 inner product */
// alphasparse_status_t alphasparse_s_dotmv(const alphasparse_operation_t transA,
//                                        const float alpha,
//                                        const alphasparse_matrix_t A,
//                                        const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                        const float *x,
//                                        const float beta,
//                                        float *y,
//                                        float *d);

// alphasparse_status_t alphasparse_d_dotmv(const alphasparse_operation_t transA,
//                                        const double alpha,
//                                        const alphasparse_matrix_t A,
//                                        const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                        const double *x,
//                                        const double beta,
//                                        double *y,
//                                        double *d);

// alphasparse_status_t alphasparse_c_dotmv(const alphasparse_operation_t transA,
//                                        const ALPHA_Complex8 alpha,
//                                        const alphasparse_matrix_t A,
//                                        const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                        const ALPHA_Complex8 *x,
//                                        const ALPHA_Complex8 beta,
//                                        ALPHA_Complex8 *y,
//                                        ALPHA_Complex8 *d);

// alphasparse_status_t alphasparse_z_dotmv(const alphasparse_operation_t transA,
//                                        const ALPHA_Complex16 alpha,
//                                        const alphasparse_matrix_t A,
//                                        const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                        const ALPHA_Complex16 *x,
//                                        const ALPHA_Complex16 beta,
//                                        ALPHA_Complex16 *y,
//                                        ALPHA_Complex16 *d);

// /*   Solves triangular system y = alpha * A^{-1} * x   */
// alphasparse_status_t alphasparse_s_trsv(const alphasparse_operation_t operation,
//                                       const float alpha,
//                                       const alphasparse_matrix_t A,
//                                       const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                       const float *x,
//                                       float *y);

// alphasparse_status_t alphasparse_d_trsv(const alphasparse_operation_t operation,
//                                       const double alpha,
//                                       const alphasparse_matrix_t A,
//                                       const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                       const double *x,
//                                       double *y);

// alphasparse_status_t alphasparse_c_trsv(const alphasparse_operation_t operation,
//                                       const ALPHA_Complex8 alpha,
//                                       const alphasparse_matrix_t A,
//                                       const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                       const ALPHA_Complex8 *x,
//                                       ALPHA_Complex8 *y);

// alphasparse_status_t alphasparse_z_trsv(const alphasparse_operation_t operation,
//                                       const ALPHA_Complex16 alpha,
//                                       const alphasparse_matrix_t A,
//                                       const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                       const ALPHA_Complex16 *x,
//                                       ALPHA_Complex16 *y);

// /*   Applies symmetric Gauss-Seidel preconditioner to symmetric system A * x = b, */
// /*   that is, it solves:                                                          */
// /*      x0       = alpha*x                                                        */
// /*      (L+D)*x1 = b - U*x0                                                       */
// /*      (D+U)*x  = b - L*x1                                                       */
// /*                                                                                */
// /*   SYMGS_MV also returns y = A*x                                                */
// alphasparse_status_t alphasparse_s_symgs(const alphasparse_operation_t op,
//                                        const alphasparse_matrix_t A,
//                                        const struct alpha_matrix_descr descr,
//                                        const float alpha,
//                                        const float *b,
//                                        float *x);

// alphasparse_status_t alphasparse_d_symgs(const alphasparse_operation_t op,
//                                        const alphasparse_matrix_t A,
//                                        const struct alpha_matrix_descr descr,
//                                        const double alpha,
//                                        const double *b,
//                                        double *x);

// alphasparse_status_t alphasparse_c_symgs(const alphasparse_operation_t op,
//                                        const alphasparse_matrix_t A,
//                                        const struct alpha_matrix_descr descr,
//                                        const ALPHA_Complex8 alpha,
//                                        const ALPHA_Complex8 *b,
//                                        ALPHA_Complex8 *x);

// alphasparse_status_t alphasparse_z_symgs(const alphasparse_operation_t op,
//                                        const alphasparse_matrix_t A,
//                                        const struct alpha_matrix_descr descr,
//                                        const ALPHA_Complex16 alpha,
//                                        const ALPHA_Complex16 *b,
//                                        ALPHA_Complex16 *x);

// alphasparse_status_t alphasparse_s_symgs_mv(const alphasparse_operation_t op,
//                                           const alphasparse_matrix_t A,
//                                           const struct alpha_matrix_descr descr,
//                                           const float alpha,
//                                           const float *b,
//                                           float *x,
//                                           float *y);

// alphasparse_status_t alphasparse_d_symgs_mv(const alphasparse_operation_t op,
//                                           const alphasparse_matrix_t A,
//                                           const struct alpha_matrix_descr descr,
//                                           const double alpha,
//                                           const double *b,
//                                           double *x,
//                                           double *y);

// alphasparse_status_t alphasparse_c_symgs_mv(const alphasparse_operation_t op,
//                                           const alphasparse_matrix_t A,
//                                           const struct alpha_matrix_descr descr,
//                                           const ALPHA_Complex8 alpha,
//                                           const ALPHA_Complex8 *b,
//                                           ALPHA_Complex8 *x,
//                                           ALPHA_Complex8 *y);

// alphasparse_status_t alphasparse_z_symgs_mv(const alphasparse_operation_t op,
//                                           const alphasparse_matrix_t A,
//                                           const struct alpha_matrix_descr descr,
//                                           const ALPHA_Complex16 alpha,
//                                           const ALPHA_Complex16 *b,
//                                           ALPHA_Complex16 *x,
//                                           ALPHA_Complex16 *y);

// /*   Computes an action of a preconditioner
//          which corresponds to the approximate matrix decomposition A ≈ (L+D)*E*(U+D)
//          for the system Ax = b.

//          L is lower triangular part of A
//          U is upper triangular part of A
//          D is diagonal values of A 
//          E is approximate diagonal inverse            
                                                                
//          That is, it solves:                                      
//              r = rhs - A*x0                                       
//              (L + D)*E*(U + D)*dx = r                             
//              x1 = x0 + dx                                        */

// alphasparse_status_t alphasparse_s_lu_smoother(const alphasparse_operation_t op,
//                                              const alphasparse_matrix_t A,
//                                              const struct alpha_matrix_descr descr,
//                                              const float *diag,
//                                              const float *approx_diag_inverse,
//                                              float *x,
//                                              const float *rhs);

// alphasparse_status_t alphasparse_d_lu_smoother(const alphasparse_operation_t op,
//                                              const alphasparse_matrix_t A,
//                                              const struct alpha_matrix_descr descr,
//                                              const double *diag,
//                                              const double *approx_diag_inverse,
//                                              double *x,
//                                              const double *rhs);

// alphasparse_status_t alphasparse_c_lu_smoother(const alphasparse_operation_t op,
//                                              const alphasparse_matrix_t A,
//                                              const struct alpha_matrix_descr descr,
//                                              const ALPHA_Complex8 *diag,
//                                              const ALPHA_Complex8 *approx_diag_inverse,
//                                              ALPHA_Complex8 *x,
//                                              const ALPHA_Complex8 *rhs);

// alphasparse_status_t alphasparse_z_lu_smoother(const alphasparse_operation_t op,
//                                              const alphasparse_matrix_t A,
//                                              const struct alpha_matrix_descr descr,
//                                              const ALPHA_Complex16 *diag,
//                                              const ALPHA_Complex16 *approx_diag_inverse,
//                                              ALPHA_Complex16 *x,
//                                              const ALPHA_Complex16 *rhs);

// /* Level 3 */

// /*   Computes y = alpha * A * x + beta * y   */
// alphasparse_status_t alphasparse_s_mm(const alphasparse_operation_t operation,
//                                     const float alpha,
//                                     const alphasparse_matrix_t A,
//                                     const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                     const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
//                                     const float *x,
//                                     const ALPHA_INT columns,
//                                     const ALPHA_INT ldx,
//                                     const float beta,
//                                     float *y,
//                                     const ALPHA_INT ldy);

// alphasparse_status_t alphasparse_d_mm(const alphasparse_operation_t operation,
//                                     const double alpha,
//                                     const alphasparse_matrix_t A,
//                                     const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                     const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
//                                     const double *x,
//                                     const ALPHA_INT columns,
//                                     const ALPHA_INT ldx,
//                                     const double beta,
//                                     double *y,
//                                     const ALPHA_INT ldy);

// alphasparse_status_t alphasparse_c_mm(const alphasparse_operation_t operation,
//                                     const ALPHA_Complex8 alpha,
//                                     const alphasparse_matrix_t A,
//                                     const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                     const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
//                                     const ALPHA_Complex8 *x,
//                                     const ALPHA_INT columns,
//                                     const ALPHA_INT ldx,
//                                     const ALPHA_Complex8 beta,
//                                     ALPHA_Complex8 *y,
//                                     const ALPHA_INT ldy);

// alphasparse_status_t alphasparse_z_mm(const alphasparse_operation_t operation,
//                                     const ALPHA_Complex16 alpha,
//                                     const alphasparse_matrix_t A,
//                                     const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                     const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
//                                     const ALPHA_Complex16 *x,
//                                     const ALPHA_INT columns,
//                                     const ALPHA_INT ldx,
//                                     const ALPHA_Complex16 beta,
//                                     ALPHA_Complex16 *y,
//                                     const ALPHA_INT ldy);

// /*   Solves triangular system y = alpha * A^{-1} * x   */
// alphasparse_status_t alphasparse_s_trsm(const alphasparse_operation_t operation,
//                                       const float alpha,
//                                       const alphasparse_matrix_t A,
//                                       const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                       const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
//                                       const float *x,
//                                       const ALPHA_INT columns,
//                                       const ALPHA_INT ldx,
//                                       float *y,
//                                       const ALPHA_INT ldy);

// alphasparse_status_t alphasparse_d_trsm(const alphasparse_operation_t operation,
//                                       const double alpha,
//                                       const alphasparse_matrix_t A,
//                                       const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                       const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
//                                       const double *x,
//                                       const ALPHA_INT columns,
//                                       const ALPHA_INT ldx,
//                                       double *y,
//                                       const ALPHA_INT ldy);

// alphasparse_status_t alphasparse_c_trsm(const alphasparse_operation_t operation,
//                                       const ALPHA_Complex8 alpha,
//                                       const alphasparse_matrix_t A,
//                                       const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                       const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
//                                       const ALPHA_Complex8 *x,
//                                       const ALPHA_INT columns,
//                                       const ALPHA_INT ldx,
//                                       ALPHA_Complex8 *y,
//                                       const ALPHA_INT ldy);

// alphasparse_status_t alphasparse_z_trsm(const alphasparse_operation_t operation,
//                                       const ALPHA_Complex16 alpha,
//                                       const alphasparse_matrix_t A,
//                                       const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
//                                       const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
//                                       const ALPHA_Complex16 *x,
//                                       const ALPHA_INT columns,
//                                       const ALPHA_INT ldx,
//                                       ALPHA_Complex16 *y,
//                                       const ALPHA_INT ldy);

// /* Sparse-sparse functionality */

// /*   Computes sum of sparse matrices: C = alpha * op(A) + B, result is sparse   */
// alphasparse_status_t alphasparse_s_add(const alphasparse_operation_t operation,
//                                      const alphasparse_matrix_t A,
//                                      const float alpha,
//                                      const alphasparse_matrix_t B,
//                                      alphasparse_matrix_t *C);

// alphasparse_status_t alphasparse_d_add(const alphasparse_operation_t operation,
//                                      const alphasparse_matrix_t A,
//                                      const double alpha,
//                                      const alphasparse_matrix_t B,
//                                      alphasparse_matrix_t *C);

// alphasparse_status_t alphasparse_c_add(const alphasparse_operation_t operation,
//                                      const alphasparse_matrix_t A,
//                                      const ALPHA_Complex8 alpha,
//                                      const alphasparse_matrix_t B,
//                                      alphasparse_matrix_t *C);

// alphasparse_status_t alphasparse_z_add(const alphasparse_operation_t operation,
//                                      const alphasparse_matrix_t A,
//                                      const ALPHA_Complex16 alpha,
//                                      const alphasparse_matrix_t B,
//                                      alphasparse_matrix_t *C);

// /*   Computes product of sparse matrices: C = op(A) * B, result is sparse   */
// alphasparse_status_t alphasparse_spmm(const alphasparse_operation_t operation,
//                                     const alphasparse_matrix_t A,
//                                     const alphasparse_matrix_t B,
//                                     alphasparse_matrix_t *C);

// /*   Computes product of sparse matrices: C = opA(A) * opB(B), result is sparse   */
// alphasparse_status_t alphasparse_sp2m(const alphasparse_operation_t transA,
//                                     const struct alpha_matrix_descr descrA,
//                                     const alphasparse_matrix_t A,
//                                     const alphasparse_operation_t transB,
//                                     const struct alpha_matrix_descr descrB,
//                                     const alphasparse_matrix_t B,
//                                     const alphasparse_request_t request,
//                                     alphasparse_matrix_t *C);

// /*   Computes product of sparse matrices: C = op(A) * (op(A))^{T for real or H for complex}, result is sparse   */
// alphasparse_status_t alphasparse_syrk(const alphasparse_operation_t operation,
//                                     const alphasparse_matrix_t A,
//                                     alphasparse_matrix_t *C);

// /*   Computes product of sparse matrices: C = op(A) * B * (op(A))^{T for real or H for complex}, result is sparse   */
// alphasparse_status_t alphasparse_sypr(const alphasparse_operation_t transA,
//                                     const alphasparse_matrix_t A,
//                                     const alphasparse_matrix_t B,
//                                     const struct alpha_matrix_descr descrB,
//                                     alphasparse_matrix_t *C,
//                                     const alphasparse_request_t request);

// /*   Computes product of sparse matrices: C = op(A) * B * (op(A))^{T for real or H for complex}, result is dense */
// alphasparse_status_t alphasparse_s_syprd(const alphasparse_operation_t op,
//                                        const alphasparse_matrix_t A,
//                                        const float *B,
//                                        const alphasparse_layout_t layoutB,
//                                        const ALPHA_INT ldb,
//                                        const float alpha,
//                                        const float beta,
//                                        float *C,
//                                        const alphasparse_layout_t layoutC,
//                                        const ALPHA_INT ldc);

// alphasparse_status_t alphasparse_d_syprd(const alphasparse_operation_t op,
//                                        const alphasparse_matrix_t A,
//                                        const double *B,
//                                        const alphasparse_layout_t layoutB,
//                                        const ALPHA_INT ldb,
//                                        const double alpha,
//                                        const double beta,
//                                        double *C,
//                                        const alphasparse_layout_t layoutC,
//                                        const ALPHA_INT ldc);

// alphasparse_status_t alphasparse_c_syprd(const alphasparse_operation_t op,
//                                        const alphasparse_matrix_t A,
//                                        const ALPHA_Complex8 *B,
//                                        const alphasparse_layout_t layoutB,
//                                        const ALPHA_INT ldb,
//                                        const ALPHA_Complex8 alpha,
//                                        const ALPHA_Complex8 beta,
//                                        ALPHA_Complex8 *C,
//                                        const alphasparse_layout_t layoutC,
//                                        const ALPHA_INT ldc);

// alphasparse_status_t alphasparse_z_syprd(const alphasparse_operation_t op,
//                                        const alphasparse_matrix_t A,
//                                        const ALPHA_Complex16 *B,
//                                        const alphasparse_layout_t layoutB,
//                                        const ALPHA_INT ldb,
//                                        const ALPHA_Complex16 alpha,
//                                        const ALPHA_Complex16 beta,
//                                        ALPHA_Complex16 *C,
//                                        const alphasparse_layout_t layoutC,
//                                        const ALPHA_INT ldc);

// /*   Computes product of sparse matrices: C = op(A) * B, result is dense   */
// alphasparse_status_t alphasparse_s_spmmd(const alphasparse_operation_t operation,
//                                        const alphasparse_matrix_t A,
//                                        const alphasparse_matrix_t B,
//                                        const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
//                                        float *C,
//                                        const ALPHA_INT ldc);

// alphasparse_status_t alphasparse_d_spmmd(const alphasparse_operation_t operation,
//                                        const alphasparse_matrix_t A,
//                                        const alphasparse_matrix_t B,
//                                        const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
//                                        double *C,
//                                        const ALPHA_INT ldc);

// alphasparse_status_t alphasparse_c_spmmd(const alphasparse_operation_t operation,
//                                        const alphasparse_matrix_t A,
//                                        const alphasparse_matrix_t B,
//                                        const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
//                                        ALPHA_Complex8 *C,
//                                        const ALPHA_INT ldc);

// alphasparse_status_t alphasparse_z_spmmd(const alphasparse_operation_t operation,
//                                        const alphasparse_matrix_t A,
//                                        const alphasparse_matrix_t B,
//                                        const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
//                                        ALPHA_Complex16 *C,
//                                        const ALPHA_INT ldc);

// /*   Computes product of sparse matrices: C = opA(A) * opB(B), result is dense*/
// alphasparse_status_t alphasparse_s_sp2md(const alphasparse_operation_t transA,
//                                        const struct alpha_matrix_descr descrA,
//                                        const alphasparse_matrix_t A,
//                                        const alphasparse_operation_t transB,
//                                        const struct alpha_matrix_descr descrB,
//                                        const alphasparse_matrix_t B,
//                                        const float alpha,
//                                        const float beta,
//                                        float *C,
//                                        const alphasparse_layout_t layout,
//                                        const ALPHA_INT ldc);

// alphasparse_status_t alphasparse_d_sp2md(const alphasparse_operation_t transA,
//                                        const struct alpha_matrix_descr descrA,
//                                        const alphasparse_matrix_t A,
//                                        const alphasparse_operation_t transB,
//                                        const struct alpha_matrix_descr descrB,
//                                        const alphasparse_matrix_t B,
//                                        const double alpha,
//                                        const double beta,
//                                        double *C,
//                                        const alphasparse_layout_t layout,
//                                        const ALPHA_INT ldc);

// alphasparse_status_t alphasparse_c_sp2md(const alphasparse_operation_t transA,
//                                        const struct alpha_matrix_descr descrA,
//                                        const alphasparse_matrix_t A,
//                                        const alphasparse_operation_t transB,
//                                        const struct alpha_matrix_descr descrB,
//                                        const alphasparse_matrix_t B,
//                                        const ALPHA_Complex8 alpha,
//                                        const ALPHA_Complex8 beta,
//                                        ALPHA_Complex8 *C,
//                                        const alphasparse_layout_t layout,
//                                        const ALPHA_INT ldc);

// alphasparse_status_t alphasparse_z_sp2md(const alphasparse_operation_t transA,
//                                        const struct alpha_matrix_descr descrA,
//                                        const alphasparse_matrix_t A,
//                                        const alphasparse_operation_t transB,
//                                        const struct alpha_matrix_descr descrB,
//                                        const alphasparse_matrix_t B,
//                                        const ALPHA_Complex16 alpha,
//                                        const ALPHA_Complex16 beta,
//                                        ALPHA_Complex16 *C,
//                                        const alphasparse_layout_t layout,
//                                        const ALPHA_INT ldc);

// /*   Computes product of sparse matrices: C = op(A) * (op(A))^{T for real or H for complex}, result is dense */
// alphasparse_status_t alphasparse_s_syrkd(const alphasparse_operation_t operation,
//                                        const alphasparse_matrix_t A,
//                                        const float alpha,
//                                        const float beta,
//                                        float *C,
//                                        const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
//                                        const ALPHA_INT ldc);

// alphasparse_status_t alphasparse_d_syrkd(const alphasparse_operation_t operation,
//                                        const alphasparse_matrix_t A,
//                                        const double alpha,
//                                        const double beta,
//                                        double *C,
//                                        const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
//                                        const ALPHA_INT ldc);

// alphasparse_status_t alphasparse_c_syrkd(const alphasparse_operation_t operation,
//                                        const alphasparse_matrix_t A,
//                                        const ALPHA_Complex8 alpha,
//                                        const ALPHA_Complex8 beta,
//                                        ALPHA_Complex8 *C,
//                                        const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
//                                        const ALPHA_INT ldc);

// alphasparse_status_t alphasparse_z_syrkd(const alphasparse_operation_t operation,
//                                        const alphasparse_matrix_t A,
//                                        const ALPHA_Complex16 alpha,
//                                        const ALPHA_Complex16 beta,
//                                        ALPHA_Complex16 *C,
//                                        const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
//                                        const ALPHA_INT ldc);

// alphasparse_status_t alphasparse_s_axpy(const ALPHA_INT nz,
//                                       const float a,
//                                       const float *x,
//                                       const ALPHA_INT *indx,
//                                       float *y);

// alphasparse_status_t alphasparse_d_axpy(const ALPHA_INT nz,
//                                       const double a,
//                                       const double *x,
//                                       const ALPHA_INT *indx,
//                                       double *y);

// alphasparse_status_t alphasparse_c_axpy(const ALPHA_INT nz,
//                                       const ALPHA_Complex8 a,
//                                       const ALPHA_Complex8 *x,
//                                       const ALPHA_INT *indx,
//                                       ALPHA_Complex8 *y);

// alphasparse_status_t alphasparse_z_axpy(const ALPHA_INT nz,
//                                       const ALPHA_Complex16 a,
//                                       const ALPHA_Complex16 *x,
//                                       const ALPHA_INT *indx,
//                                       ALPHA_Complex16 *y);

// alphasparse_status_t alphasparse_s_gthr(const ALPHA_INT nz,
//                                       const float *y,
//                                       float *x,
//                                       const ALPHA_INT *indx);

// alphasparse_status_t alphasparse_d_gthr(const ALPHA_INT nz,
//                                       const double *y,
//                                       double *x,
//                                       const ALPHA_INT *indx);

// alphasparse_status_t alphasparse_c_gthr(const ALPHA_INT nz,
//                                       const ALPHA_Complex8 *y,
//                                       ALPHA_Complex8 *x,
//                                       const ALPHA_INT *indx);

// alphasparse_status_t alphasparse_z_gthr(const ALPHA_INT nz,
//                                       const ALPHA_Complex16 *y,
//                                       ALPHA_Complex16 *x,
//                                       const ALPHA_INT *indx);

// alphasparse_status_t alphasparse_s_gthrz(const ALPHA_INT nz,
//                                        float *y,
//                                        float *x,
//                                        const ALPHA_INT *indx);

// alphasparse_status_t alphasparse_d_gthrz(const ALPHA_INT nz,
//                                        double *y,
//                                        double *x,
//                                        const ALPHA_INT *indx);

// alphasparse_status_t alphasparse_c_gthrz(const ALPHA_INT nz,
//                                        ALPHA_Complex8 *y,
//                                        ALPHA_Complex8 *x,
//                                        const ALPHA_INT *indx);

// alphasparse_status_t alphasparse_z_gthrz(const ALPHA_INT nz,
//                                        ALPHA_Complex16 *y,
//                                        ALPHA_Complex16 *x,
//                                        const ALPHA_INT *indx);

// alphasparse_status_t alphasparse_s_rot(const ALPHA_INT nz,
//                                      float *x,
//                                      const ALPHA_INT *indx,
//                                      float *y,
//                                      const float c,
//                                      const float s);

// alphasparse_status_t alphasparse_d_rot(const ALPHA_INT nz,
//                                      double *x,
//                                      const ALPHA_INT *indx,
//                                      double *y,
//                                      const double c,
//                                      const double s);

// alphasparse_status_t alphasparse_s_sctr(const ALPHA_INT nz,
//                                       const float *x,
//                                       const ALPHA_INT *indx,
//                                       float *y);

// alphasparse_status_t alphasparse_d_sctr(const ALPHA_INT nz,
//                                       const double *x,
//                                       const ALPHA_INT *indx,
//                                       double *y);

// alphasparse_status_t alphasparse_c_sctr(const ALPHA_INT nz,
//                                       const ALPHA_Complex8 *x,
//                                       const ALPHA_INT *indx,
//                                       ALPHA_Complex8 *y);

// alphasparse_status_t alphasparse_z_sctr(const ALPHA_INT nz,
//                                       const ALPHA_Complex16 *x,
//                                       const ALPHA_INT *indx,
//                                       ALPHA_Complex16 *y);

// float alphasparse_s_doti(const ALPHA_INT nz,
//                         const float *x,
//                         const ALPHA_INT *indx,
//                         const float *y);

// double alphasparse_d_doti(const ALPHA_INT nz,
//                          const double *x,
//                          const ALPHA_INT *indx,
//                          const double *y);

// void alphasparse_c_dotci_sub(const ALPHA_INT nz,
//                             const ALPHA_Complex8 *x,
//                             const ALPHA_INT *indx,
//                             const ALPHA_Complex8 *y,
//                             ALPHA_Complex8 *dutci);

// void alphasparse_z_dotci_sub(const ALPHA_INT nz,
//                             const ALPHA_Complex16 *x,
//                             const ALPHA_INT *indx,
//                             const ALPHA_Complex16 *y,
//                             ALPHA_Complex16 *dutci);

// void alphasparse_c_dotui_sub(const ALPHA_INT nz,
//                             const ALPHA_Complex8 *x,
//                             const ALPHA_INT *indx,
//                             const ALPHA_Complex8 *y,
//                             ALPHA_Complex8 *dutui);

// void alphasparse_z_dotui_sub(const ALPHA_INT nz,
//                             const ALPHA_Complex16 *x,
//                             const ALPHA_INT *indx,
//                             const ALPHA_Complex16 *y,
//                             ALPHA_Complex16 *dutui);
