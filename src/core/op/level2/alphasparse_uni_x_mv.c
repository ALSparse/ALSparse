#include "alphasparse.h"

alphasparse_status_t ONAME(const alphasparse_operation_t operation,
                          const ALPHA_Number alpha,
                          const alphasparse_matrix_t A,
                          const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                          const ALPHA_Number *x,
                          const ALPHA_Number beta,
                          ALPHA_Number *y)
{
    check_null_return(A, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(x, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(y, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_return(A->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);

#ifndef COMPLEX
    if(operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
#endif

    if(descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC || descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        // check if it is a square matrix 
        check_return(!check_equal_row_col(A),ALPHA_SPARSE_STATUS_INVALID_VALUE);


    if (A->exe == ALPHA_SPARSE_EXE_HOST)
    {
#ifdef S
        alphasparse_s_mv(operation, alpha, A, descr, x, beta, y);
#endif
#ifdef D
        alphasparse_d_mv(operation, alpha, A, descr, x, beta, y);
#endif
#ifdef C
        alphasparse_c_mv(operation, alpha, A, descr, x, beta, y);
#endif
#ifdef Z
        alphasparse_z_mv(operation, alpha, A, descr, x, beta, y);
#endif
    }
    else if (A->exe == ALPHA_SPARSE_EXE_DEVICE)
    {
#ifdef __DCU__
        alpha_dcu_matrix_descr_t descrA;
        alphasparse_dcu_create_mat_descr(&descrA);
        alphasparse_dcu_mat_info_t info         = (alphasparse_dcu_mat_info_t)alpha_malloc(sizeof(struct _alphasparse_dcu_mat_info));
        alphasparse_dcu_csrmv_info_t csrmv_info  = (alphasparse_dcu_csrmv_info_t)alpha_malloc(sizeof(struct _alphasparse_dcu_csrmv_info));
        info->csrmv_info                         = csrmv_info;
        info->csrmv_info->csr_adaptive_has_tuned = false;
        ALPHA_INT m   = A->mat->rows;
        ALPHA_INT n   = A->mat->cols;
        ALPHA_INT nnz = A->mat->rows_end[m - 1] - A->mat->rows_start[0] + 1
        if (A->format == ALPHA_SPARSE_FORMAT_CSR)
        {
#ifdef S
            alphasparse_dcu_s_csrmv(A->handle, operation, m, n, nnz, &alpha, descrA, A->mat->d_values, A->mat->d_row_ptr, A->mat->d_col_indx, info, x, &beta, y);
#endif
#ifdef D
            alphasparse_dcu_d_csrmv(A->handle, operation, m, n, nnz, &alpha, descrA, A->mat->d_values, A->mat->d_row_ptr, A->mat->d_col_indx, info, x, &beta, y);
#endif
#ifdef C
            alphasparse_dcu_c_csrmv(A->handle, operation, m, n, nnz, &alpha, descrA, A->mat->d_values, A->mat->d_row_ptr, A->mat->d_col_indx, info, x, &beta, y);
#endif
#ifdef Z
            alphasparse_dcu_z_csrmv(A->handle, operation, m, n, nnz, &alpha, descrA, A->mat->d_values, A->mat->d_row_ptr, A->mat->d_col_indx, info, x, &beta, y);
#endif
        }
#endif
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}
