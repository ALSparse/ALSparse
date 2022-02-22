#include "alphasparse/handle.h"
#include "alphasparse/spapi_dcu.h"
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

static alphasparse_status_t (*trsv_bsr_diag_fill_operation[])(alphasparse_dcu_handle_t handle,
                                                              alphasparse_layout_t dir,
                                                              ALPHA_INT mb,
                                                              ALPHA_INT nnzb,
                                                              const ALPHA_Number alpha,
                                                              const ALPHA_Number *bsr_val,
                                                              const ALPHA_INT *bsr_row_ptr,
                                                              const ALPHA_INT *bsr_col_ind,
                                                              ALPHA_INT bsr_dim,
                                                              alphasparse_dcu_mat_info_t info,
                                                              const ALPHA_Number *x,
                                                              ALPHA_Number *y,
                                                              alphasparse_dcu_solve_policy_t policy,
                                                              void *temp_buffer) = {
    dcu_trsv_bsr_n_lo,
    dcu_trsv_bsr_u_lo,
    dcu_trsv_bsr_n_hi,
    dcu_trsv_bsr_u_hi,
    dcu_trsv_bsr_n_lo_trans,
    dcu_trsv_bsr_u_lo_trans,
    dcu_trsv_bsr_n_hi_trans,
    dcu_trsv_bsr_u_hi_trans,
#ifdef COMPLEX
    NULL, // dcu_trsv_bsr_n_lo_conj, //nerver implement
    NULL, // dcu_trsv_bsr_u_lo_conj, //nerver implement
    NULL, // dcu_trsv_bsr_n_hi_conj, //nerver implement
    NULL // dcu_trsv_bsr_u_hi_conj, //nerver implement
#endif
};

alphasparse_status_t ONAME(alphasparse_dcu_handle_t handle,
                           alphasparse_layout_t dir,
                           alphasparse_operation_t trans,
                           ALPHA_INT mb,
                           ALPHA_INT nnzb,
                           const ALPHA_Number *alpha,
                           const alpha_dcu_matrix_descr_t descr,
                           const ALPHA_Number *bsr_val,
                           const ALPHA_INT *bsr_row_ptr,
                           const ALPHA_INT *bsr_col_ind,
                           ALPHA_INT bsr_dim,
                           alphasparse_dcu_mat_info_t info,
                           const ALPHA_Number *x,
                           ALPHA_Number *y,
                           alphasparse_dcu_solve_policy_t policy,
                           void *temp_buffer)
{
    // Check for valid handle and matrix descriptor
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

    // Check direction
    if (dir != ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR && dir != ALPHA_SPARSE_LAYOUT_ROW_MAJOR) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Check sizes
    if (mb < 0 || nnzb < 0 || bsr_dim < 0) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return if possible
    if (mb == 0 || nnzb == 0 || bsr_dim == 0) {
        return ALPHA_SPARSE_STATUS_SUCCESS;
    }

    // Check pointer arguments
    if (alpha == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Check pointer arguments
    if (bsr_val == nullptr || bsr_row_ptr == nullptr || bsr_col_ind == nullptr || alpha == nullptr || x == nullptr || y == nullptr) // || temp_buffer == nullptr)
    {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    ALPHA_Number l_alpha = *alpha;
    if (descr->type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR) {
        check_null_return(trsv_bsr_diag_fill_operation[index3(trans, descr->mode, descr->diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
        return trsv_bsr_diag_fill_operation[index3(trans, descr->mode, descr->diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](handle,
                                                                                                                                             dir,
                                                                                                                                             mb,
                                                                                                                                             nnzb,
                                                                                                                                             l_alpha,
                                                                                                                                             bsr_val,
                                                                                                                                             bsr_row_ptr,
                                                                                                                                             bsr_col_ind,
                                                                                                                                             bsr_dim,
                                                                                                                                             info,
                                                                                                                                             x,
                                                                                                                                             y,
                                                                                                                                             policy,
                                                                                                                                             temp_buffer);
    } else if (descr->type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL) {
        // todo
        // check_null_return(diagsv_bsr_diag[descr->diag], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
        // return diagsv_bsr_diag[descr->diag](alpha, A->mat, x, y);
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    } else {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
