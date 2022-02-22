/**
 * @brief implement for alphasparse_?_mv intelface
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "alphasparse/handle.h"
#include "alphasparse/spapi_dcu.h"
#include "alphasparse/kernel.h"
#include "alphasparse/spapi.h"
#include "alphasparse/util.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

alphasparse_status_t ONAME(alphasparse_dcu_handle_t handle,
                           const alphasparse_operation_t operation,
                           const ALPHA_Number* alpha,
                           const alphasparse_matrix_t A,
                           const alpha_dcu_matrix_descr_t descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                           const ALPHA_Number* x,
                           const ALPHA_Number* beta,
                           ALPHA_Number* y)
{
    check_null_return(A, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(x, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(y, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_return(A->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);

    alphasparse_status_t st = ALPHA_SPARSE_STATUS_INVALID_VALUE;

    if (A->format == ALPHA_SPARSE_FORMAT_CSR) {
        ALPHA_SPMAT_CSR* mat = (ALPHA_SPMAT_CSR*)A->mat;

        if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
            st =
                alphasparse_dcu_s_csrmv(handle,
                                        operation,
                                        mat->rows,
                                        mat->cols,
                                        mat->rows_end[mat->rows - 1],
                                        (float*)alpha,
                                        descr,
                                        (float*)mat->d_values,
                                        mat->d_row_ptr,
                                        mat->d_col_indx,
                                        (alphasparse_dcu_mat_info_t)A->dcu_info,
                                        (float*)x,
                                        (float*)beta,
                                        (float*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
            st =
                alphasparse_dcu_d_csrmv(handle,
                                        operation,
                                        mat->rows,
                                        mat->cols,
                                        mat->rows_end[mat->rows - 1],
                                        (double*)alpha,
                                        descr,
                                        (double*)mat->d_values,
                                        mat->d_row_ptr,
                                        mat->d_col_indx,
                                        (alphasparse_dcu_mat_info_t)A->dcu_info,
                                        (double*)x,
                                        (double*)beta,
                                        (double*)y);

        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
            st =
                alphasparse_dcu_c_csrmv(handle,
                                        operation,
                                        mat->rows,
                                        mat->cols,
                                        mat->rows_end[mat->rows - 1],
                                        (ALPHA_Complex8*)alpha,
                                        descr,
                                        (ALPHA_Complex8*)mat->d_values,
                                        mat->d_row_ptr,
                                        mat->d_col_indx,
                                        (alphasparse_dcu_mat_info_t)A->dcu_info,
                                        (ALPHA_Complex8*)x,
                                        (ALPHA_Complex8*)beta,
                                        (ALPHA_Complex8*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
            st =
                alphasparse_dcu_z_csrmv(handle,
                                        operation,
                                        mat->rows,
                                        mat->cols,
                                        mat->rows_end[mat->rows - 1],
                                        (ALPHA_Complex16*)alpha,
                                        descr,
                                        (ALPHA_Complex16*)mat->d_values,
                                        mat->d_row_ptr,
                                        mat->d_col_indx,
                                        (alphasparse_dcu_mat_info_t)A->dcu_info,
                                        (ALPHA_Complex16*)x,
                                        (ALPHA_Complex16*)beta,
                                        (ALPHA_Complex16*)y);
        } else {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    } else if (A->format == ALPHA_SPARSE_FORMAT_COO) {
        ALPHA_SPMAT_COO* mat = (ALPHA_SPMAT_COO*)A->mat;
        if (!mat->ordered) {
            printf("alphasparse_dcu_x_coomv need sorted coo format.\n");
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }

        if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
            st =
                alphasparse_dcu_s_coomv(handle,
                                        operation,
                                        mat->rows,
                                        mat->cols,
                                        mat->nnz,
                                        (float*)alpha,
                                        descr,
                                        (float*)mat->d_values,
                                        mat->d_rows_indx,
                                        mat->d_cols_indx,
                                        (float*)x,
                                        (float*)beta,
                                        (float*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
            st =
                alphasparse_dcu_d_coomv(handle,
                                        operation,
                                        mat->rows,
                                        mat->cols,
                                        mat->nnz,
                                        (double*)alpha,
                                        descr,
                                        (double*)mat->d_values,
                                        mat->d_rows_indx,
                                        mat->d_cols_indx,
                                        (double*)x,
                                        (double*)beta,
                                        (double*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
            st =
                alphasparse_dcu_c_coomv(handle,
                                        operation,
                                        mat->rows,
                                        mat->cols,
                                        mat->nnz,
                                        (ALPHA_Complex8*)alpha,
                                        descr,
                                        (ALPHA_Complex8*)mat->d_values,
                                        mat->d_rows_indx,
                                        mat->d_cols_indx,
                                        (ALPHA_Complex8*)x,
                                        (ALPHA_Complex8*)beta,
                                        (ALPHA_Complex8*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
            st =
                alphasparse_dcu_z_coomv(handle,
                                        operation,
                                        mat->rows,
                                        mat->cols,
                                        mat->nnz,
                                        (ALPHA_Complex16*)alpha,
                                        descr,
                                        (ALPHA_Complex16*)mat->d_values,
                                        mat->d_rows_indx,
                                        mat->d_cols_indx,
                                        (ALPHA_Complex16*)x,
                                        (ALPHA_Complex16*)beta,
                                        (ALPHA_Complex16*)y);
        } else {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    } else if (A->format == ALPHA_SPARSE_FORMAT_BSR) {
        ALPHA_SPMAT_BSR* mat = (ALPHA_SPMAT_BSR*)A->mat;

        if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
            st =
                alphasparse_dcu_s_bsrmv(handle,
                                        mat->block_layout,
                                        operation,
                                        mat->rows,
                                        mat->cols,
                                        mat->rows_end[mat->rows - 1],
                                        (float*)alpha,
                                        descr,
                                        (float*)mat->d_values,
                                        mat->d_rows_ptr,
                                        mat->d_col_indx,
                                        mat->block_size,
                                        (float*)x,
                                        (float*)beta,
                                        (float*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
            st =
                alphasparse_dcu_d_bsrmv(handle,
                                        mat->block_layout,
                                        operation,
                                        mat->rows,
                                        mat->cols,
                                        mat->rows_end[mat->rows - 1],
                                        (double*)alpha,
                                        descr,
                                        (double*)mat->d_values,
                                        mat->d_rows_ptr,
                                        mat->d_col_indx,
                                        mat->block_size,
                                        (double*)x,
                                        (double*)beta,
                                        (double*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
            st =
                alphasparse_dcu_c_bsrmv(handle,
                                        mat->block_layout,
                                        operation,
                                        mat->rows,
                                        mat->cols,
                                        mat->rows_end[mat->rows - 1],
                                        (ALPHA_Complex8*)alpha,
                                        descr,
                                        (ALPHA_Complex8*)mat->d_values,
                                        mat->d_rows_ptr,
                                        mat->d_col_indx,
                                        mat->block_size,
                                        (ALPHA_Complex8*)x,
                                        (ALPHA_Complex8*)beta,
                                        (ALPHA_Complex8*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
            st =
                alphasparse_dcu_z_bsrmv(handle,
                                        mat->block_layout,
                                        operation,
                                        mat->rows,
                                        mat->cols,
                                        mat->rows_end[mat->rows - 1],
                                        (ALPHA_Complex16*)alpha,
                                        descr,
                                        (ALPHA_Complex16*)mat->d_values,
                                        mat->d_rows_ptr,
                                        mat->d_col_indx,
                                        mat->block_size,
                                        (ALPHA_Complex16*)x,
                                        (ALPHA_Complex16*)beta,
                                        (ALPHA_Complex16*)y);
        } else {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    } else if (A->format == ALPHA_SPARSE_FORMAT_GEBSR) {
        ALPHA_SPMAT_GEBSR* mat = (ALPHA_SPMAT_GEBSR*)A->mat;

        if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
            st =
                alphasparse_dcu_s_gebsrmv(handle,
                                          mat->block_layout,
                                          operation,
                                          mat->rows,
                                          mat->cols,
                                          mat->rows_end[mat->rows - 1],
                                          (float*)alpha,
                                          descr,
                                          (float*)mat->d_values,
                                          mat->d_rows_ptr,
                                          mat->d_col_indx,
                                          mat->row_block_dim,
                                          mat->col_block_dim,
                                          (float*)x,
                                          (float*)beta,
                                          (float*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
            st =
                alphasparse_dcu_d_gebsrmv(handle,
                                          mat->block_layout,
                                          operation,
                                          mat->rows,
                                          mat->cols,
                                          mat->rows_end[mat->rows - 1],
                                          (double*)alpha,
                                          descr,
                                          (double*)mat->d_values,
                                          mat->d_rows_ptr,
                                          mat->d_col_indx,
                                          mat->row_block_dim,
                                          mat->col_block_dim,
                                          (double*)x,
                                          (double*)beta,
                                          (double*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
            st =
                alphasparse_dcu_c_gebsrmv(handle,
                                          mat->block_layout,
                                          operation,
                                          mat->rows,
                                          mat->cols,
                                          mat->rows_end[mat->rows - 1],
                                          (ALPHA_Complex8*)alpha,
                                          descr,
                                          (ALPHA_Complex8*)mat->d_values,
                                          mat->d_rows_ptr,
                                          mat->d_col_indx,
                                          mat->row_block_dim,
                                          mat->col_block_dim,
                                          (ALPHA_Complex8*)x,
                                          (ALPHA_Complex8*)beta,
                                          (ALPHA_Complex8*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
            st =
                alphasparse_dcu_z_gebsrmv(handle,
                                          mat->block_layout,
                                          operation,
                                          mat->rows,
                                          mat->cols,
                                          mat->rows_end[mat->rows - 1],
                                          (ALPHA_Complex16*)alpha,
                                          descr,
                                          (ALPHA_Complex16*)mat->d_values,
                                          mat->d_rows_ptr,
                                          mat->d_col_indx,
                                          mat->row_block_dim,
                                          mat->col_block_dim,
                                          (ALPHA_Complex16*)x,
                                          (ALPHA_Complex16*)beta,
                                          (ALPHA_Complex16*)y);
        } else {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    } else if (A->format == ALPHA_SPARSE_FORMAT_ELL) {
        ALPHA_SPMAT_ELL* mat = (ALPHA_SPMAT_ELL*)A->mat;

        if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
            st =
                alphasparse_dcu_s_ellmv(handle,
                                        operation,
                                        mat->rows,
                                        mat->cols,
                                        (float*)alpha,
                                        descr,
                                        (float*)mat->d_values,
                                        mat->d_indices,
                                        mat->ld,
                                        (float*)x,
                                        (float*)beta,
                                        (float*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
            st =
                alphasparse_dcu_d_ellmv(handle,
                                        operation,
                                        mat->rows,
                                        mat->cols,
                                        (double*)alpha,
                                        descr,
                                        (double*)mat->d_values,
                                        mat->d_indices,
                                        mat->ld,
                                        (double*)x,
                                        (double*)beta,
                                        (double*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
            st =
                alphasparse_dcu_c_ellmv(handle,
                                        operation,
                                        mat->rows,
                                        mat->cols,
                                        (ALPHA_Complex8*)alpha,
                                        descr,
                                        (ALPHA_Complex8*)mat->d_values,
                                        mat->d_indices,
                                        mat->ld,
                                        (ALPHA_Complex8*)x,
                                        (ALPHA_Complex8*)beta,
                                        (ALPHA_Complex8*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
            st =
                alphasparse_dcu_z_ellmv(handle,
                                        operation,
                                        mat->rows,
                                        mat->cols,
                                        (ALPHA_Complex16*)alpha,
                                        descr,
                                        (ALPHA_Complex16*)mat->d_values,
                                        mat->d_indices,
                                        mat->ld,
                                        (ALPHA_Complex16*)x,
                                        (ALPHA_Complex16*)beta,
                                        (ALPHA_Complex16*)y);
        } else {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    } else if (A->format == ALPHA_SPARSE_FORMAT_HYB) {
        ALPHA_SPMAT_HYB* mat = (ALPHA_SPMAT_HYB*)A->mat;

        if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
            st =
                alphasparse_dcu_s_hybmv(handle,
                                        operation,
                                        (float*)alpha,
                                        descr,
                                        (spmat_hyb_s_t*)mat,
                                        (float*)x,
                                        (float*)beta,
                                        (float*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
            st =
                alphasparse_dcu_d_hybmv(handle,
                                        operation,
                                        (double*)alpha,
                                        descr,
                                        (spmat_hyb_d_t*)mat,
                                        (double*)x,
                                        (double*)beta,
                                        (double*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
            st =
                alphasparse_dcu_c_hybmv(handle,
                                        operation,
                                        (ALPHA_Complex8*)alpha,
                                        descr,
                                        (spmat_hyb_c_t*)mat,
                                        (ALPHA_Complex8*)x,
                                        (ALPHA_Complex8*)beta,
                                        (ALPHA_Complex8*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
            st =
                alphasparse_dcu_z_hybmv(handle,
                                        operation,
                                        (ALPHA_Complex16*)alpha,
                                        descr,
                                        (spmat_hyb_z_t*)mat,
                                        (ALPHA_Complex16*)x,
                                        (ALPHA_Complex16*)beta,
                                        (ALPHA_Complex16*)y);
        } else {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    } else if (A->format == ALPHA_SPARSE_FORMAT_CSR5) {
        ALPHA_SPMAT_CSR5* mat = (ALPHA_SPMAT_CSR5*)A->mat;

        if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
            st = alphasparse_dcu_s_csr5mv(handle, operation, (float*)alpha, descr, (spmat_csr5_s_t*)mat, nullptr, (float*)x, (float*)beta, (float*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
            st = alphasparse_dcu_d_csr5mv(handle, operation, (double*)alpha, descr, (spmat_csr5_d_t*)mat, nullptr, (double*)x, (double*)beta, (double*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
            st = alphasparse_dcu_c_csr5mv(handle, operation, (ALPHA_Complex8*)alpha, descr, (spmat_csr5_c_t*)mat, nullptr, (ALPHA_Complex8*)x, (ALPHA_Complex8*)beta, (ALPHA_Complex8*)y);
        } else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
            st = alphasparse_dcu_z_csr5mv(handle, operation, (ALPHA_Complex16*)alpha, descr, (spmat_csr5_z_t*)mat, nullptr, (ALPHA_Complex16*)x, (ALPHA_Complex16*)beta, (ALPHA_Complex16*)y);
        } else {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    } else {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    return st;
}

#ifdef __cplusplus
}
#endif /* __cplusplus */
