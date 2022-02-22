#include "alphasparse/handle.h"
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "alphasparse/spapi_dcu.h"
#include "alphasparse/kernel_dcu.h"
#include "alphasparse/util/error.h"
#include "alphasparse/util_dcu.h"
#include "alphasparse/compute.h"
#include "alphasparse/opt.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/util/check.h"

static alphasparse_status_t (*trsv_csr_diag_fill_operation[])(alphasparse_dcu_handle_t handle,
                                                              ALPHA_INT m,
                                                              ALPHA_INT nnz,
                                                              const ALPHA_Number alpha,
                                                              const ALPHA_Number *csr_val,
                                                              const ALPHA_INT *csr_row_ptr,
                                                              const ALPHA_INT *csr_col_ind,
                                                              alphasparse_dcu_mat_info_t info,
                                                              const ALPHA_Number *x,
                                                              ALPHA_Number *y,
                                                              alphasparse_dcu_solve_policy_t policy,
                                                              void *temp_buffer) = {
    dcu_trsv_csr_n_lo,
    dcu_trsv_csr_u_lo,
    dcu_trsv_csr_n_hi,
    dcu_trsv_csr_u_hi,
    dcu_trsv_csr_n_lo_trans,
    dcu_trsv_csr_u_lo_trans,
    dcu_trsv_csr_n_hi_trans,
    dcu_trsv_csr_u_hi_trans,
#ifdef COMPLEX
    NULL, // dcu_trsv_csr_n_lo_conj, //nerver implement
    NULL, // dcu_trsv_csr_u_lo_conj, //nerver implement
    NULL, // dcu_trsv_csr_n_hi_conj, //nerver implement
    NULL // dcu_trsv_csr_u_hi_conj, //nerver implement
#endif
};

static alphasparse_status_t (*diagsv_csr_diag[])(alphasparse_dcu_handle_t handle,
                                                 ALPHA_INT m,
                                                 ALPHA_INT nnz,
                                                 const ALPHA_Number alpha,
                                                 const ALPHA_Number *csr_val,
                                                 const ALPHA_INT *csr_row_ptr,
                                                 const ALPHA_INT *csr_col_ind,
                                                 alphasparse_dcu_mat_info_t info,
                                                 const ALPHA_Number *x,
                                                 ALPHA_Number *y,
                                                 alphasparse_dcu_solve_policy_t policy,
                                                 void *temp_buffer) = {
    dcu_diagsv_csr_n,
    dcu_diagsv_csr_u,
};

alphasparse_status_t ONAME(alphasparse_dcu_handle_t handle,
                           alphasparse_operation_t trans,
                           ALPHA_INT m,
                           ALPHA_INT nnz,
                           const ALPHA_Number *alpha,
                           const alpha_dcu_matrix_descr_t descr,
                           const ALPHA_Number *csr_val,
                           const ALPHA_INT *csr_row_ptr,
                           const ALPHA_INT *csr_col_ind,
                           alphasparse_dcu_mat_info_t info,
                           const ALPHA_Number *x,
                           ALPHA_Number *y,
                           alphasparse_dcu_solve_policy_t policy,
                           void *temp_buffer)
{
    // Check for valid handle and matrix descriptor
    alphasparse_operation_t operation = trans;
    if (handle == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
    }

    if (descr == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }
    // Check index base
    if (descr->base != ALPHA_SPARSE_INDEX_BASE_ZERO && descr->base != ALPHA_SPARSE_INDEX_BASE_ONE) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    // Check sizes
    if (m < 0 || nnz < 0) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return if possible
    if (m == 0 || nnz == 0) {
        return ALPHA_SPARSE_STATUS_SUCCESS;
    }

    // Check pointer arguments
    if (alpha == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    //
    // Check the rest of pointer arguments
    //
    if (csr_val == nullptr || csr_row_ptr == nullptr || csr_col_ind == nullptr || x == nullptr || y == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    ALPHA_Number l_alpha = *alpha;
    if (descr->type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR) {
        check_null_return(trsv_csr_diag_fill_operation[index3(operation, descr->mode, descr->diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
        return trsv_csr_diag_fill_operation[index3(operation, descr->mode, descr->diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](handle,
                                                                                                                                                 m,
                                                                                                                                                 nnz,
                                                                                                                                                 l_alpha,
                                                                                                                                                 csr_val,
                                                                                                                                                 csr_row_ptr,
                                                                                                                                                 csr_col_ind,
                                                                                                                                                 info,
                                                                                                                                                 x,
                                                                                                                                                 y,
                                                                                                                                                 policy,
                                                                                                                                                 temp_buffer);
    } else if (descr->type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL) {
        // todo
        // check_null_return(diagsv_csr_diag[descr->diag], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
        // return diagsv_csr_diag[descr->diag](alpha, A->mat, x, y);
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    } else {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
