#include "alphasparse/spapi.h"
#ifdef __DCU__
#include "alphasparse_dcu.h"
#endif
#include "alphasparse/format.h"
#include "alphasparse/spmat.h"

alphasparse_status_t ONAME(const alphasparse_operation_t operation,
                          const ALPHA_Number alpha,
                          const alphasparse_matrix_t A,
                          const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                          const ALPHA_Number *x,
                          const ALPHA_Number beta,
                          ALPHA_Number *y)
{
    alphasparse_status_t status;
    if (A->exe == ALPHA_SPARSE_EXE_HOST)
    {
#ifdef S
        status = alphasparse_s_mv(operation, alpha, A, descr, x, beta, y);
#endif
#ifdef D
        status = alphasparse_d_mv(operation, alpha, A, descr, x, beta, y);
#endif
#ifdef C
        status = alphasparse_c_mv(operation, alpha, A, descr, x, beta, y);
#endif
#ifdef Z
        status = alphasparse_z_mv(operation, alpha, A, descr, x, beta, y);
#endif
        return status ;
    }
    else if (A->exe == ALPHA_SPARSE_EXE_DEVICE)
    {
#ifdef __DCU__
        alpha_dcu_matrix_descr_t descrA;
        alphasparse_dcu_create_mat_descr(&descrA);
        alphasparse_dcu_handle_t handle;
        init_handle(&handle);
        alphasparse_dcu_get_handle(&handle);
        alphasparse_dcu_mat_info_t info          = (alphasparse_dcu_mat_info_t)alpha_malloc(sizeof(struct _alphasparse_dcu_mat_info));
        alphasparse_dcu_csrmv_info_t csrmv_info  = (alphasparse_dcu_csrmv_info_t)alpha_malloc(sizeof(struct _alphasparse_dcu_csrmv_info));
        info->csrmv_info                         = csrmv_info;
        info->csrmv_info->csr_adaptive_has_tuned = false;
        
        if (A->format == ALPHA_SPARSE_FORMAT_CSR)
        {
            ALPHA_SPMAT_CSR* AA = (ALPHA_SPMAT_CSR*)((alphasparse_matrix*)A)->mat;
            ALPHA_INT m   = AA->rows;
            ALPHA_INT n   = AA->cols;
            ALPHA_INT nnz = AA->rows_end[m - 1] - AA->rows_start[0];
#ifdef S            
            status = alphasparse_dcu_s_csrmv(handle, operation, m, n, nnz, &alpha, descrA, AA->d_values, AA->d_row_ptr, AA->d_col_indx, info, x, &beta, y);
#endif
#ifdef D
            status = alphasparse_dcu_d_csrmv(handle, operation, m, n, nnz, &alpha, descrA, AA->d_values, AA->d_row_ptr, AA->d_col_indx, info, x, &beta, y);
#endif
#ifdef C
            status = alphasparse_dcu_c_csrmv(handle, operation, m, n, nnz, &alpha, descrA, AA->d_values, AA->d_row_ptr, AA->d_col_indx, info, x, &beta, y);
#endif
#ifdef Z
            status = alphasparse_dcu_z_csrmv(handle, operation, m, n, nnz, &alpha, descrA, AA->d_values, AA->d_row_ptr, AA->d_col_indx, info, x, &beta, y);
#endif
            alphasparse_dcu_destory_handle(handle);
            return status ;
        }
#endif
        return status ;
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}
