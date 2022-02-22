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
#include <memory.h>
#include "assert.h"

static alphasparse_status_t (*trsm_csr_diag_fill_operation[])(alphasparse_dcu_handle_t handle,
                                                              ALPHA_INT m,
                                                              ALPHA_INT nrhs,
                                                              ALPHA_INT nnz,
                                                              const ALPHA_Number alpha,
                                                              const ALPHA_Number *csr_val,
                                                              const ALPHA_INT *csr_row_ptr,
                                                              const ALPHA_INT *csr_col_ind,
                                                              ALPHA_Number *B,
                                                              ALPHA_INT ldb,
                                                              alphasparse_dcu_mat_info_t info,
                                                              alphasparse_dcu_solve_policy_t policy,
                                                              void *temp_buffer) = {
    dcu_trsm_csr_n_lo,
    dcu_trsm_csr_u_lo,
    dcu_trsm_csr_n_hi,
    dcu_trsm_csr_u_hi,
    dcu_trsm_csr_n_lo_transA,
    dcu_trsm_csr_u_lo_transA,
    dcu_trsm_csr_n_hi_transA,
    dcu_trsm_csr_u_hi_transA,
    NULL, // dcu_trsm_csr_n_lo_conjA, // nerver implement
    NULL, // dcu_trsm_csr_u_lo_conjA, // nerver implement
    NULL, // dcu_trsm_csr_n_hi_conjA, // nerver implement
    NULL, // dcu_trsm_csr_u_hi_conjA, // nerver implement
    dcu_trsm_csr_n_lo_transB,
    dcu_trsm_csr_u_lo_transB,
    dcu_trsm_csr_n_hi_transB,
    dcu_trsm_csr_u_hi_transB,
    dcu_trsm_csr_n_lo_transAB,
    dcu_trsm_csr_u_lo_transAB,
    dcu_trsm_csr_n_hi_transAB,
    dcu_trsm_csr_u_hi_transAB,
    NULL, // dcu_trsm_csr_n_lo_conjAtransB, // nerver implement
    NULL, // dcu_trsm_csr_u_lo_conjAtransB, // nerver implement
    NULL, // dcu_trsm_csr_n_hi_conjAtransB, // nerver implement
    NULL, // dcu_trsm_csr_u_hi_conjAtransB, // nerver implement
    NULL, // dcu_trsm_csr_n_lo_conjB,       // nerver implement
    NULL, // dcu_trsm_csr_u_lo_conjB,       // nerver implement
    NULL, // dcu_trsm_csr_n_hi_conjB,       // nerver implement
    NULL, // dcu_trsm_csr_u_hi_conjB,       // nerver implement
    NULL, // dcu_trsm_csr_n_lo_transAconjB, // nerver implement
    NULL, // dcu_trsm_csr_u_lo_transAconjB, // nerver implement
    NULL, // dcu_trsm_csr_n_hi_transAconjB, // nerver implement
    NULL, // dcu_trsm_csr_u_hi_transAconjB, // nerver implement
    NULL, // dcu_trsm_csr_n_lo_conjAB,      // nerver implement
    NULL, // dcu_trsm_csr_u_lo_conjAB,      // nerver implement
    NULL, // dcu_trsm_csr_n_hi_conjAB,      // nerver implement
    NULL // dcu_trsm_csr_u_hi_conjAB,      // nerver implement
};

alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      alphasparse_operation_t trans_A,
      alphasparse_operation_t trans_B,
      ALPHA_INT m,
      ALPHA_INT nrhs,
      ALPHA_INT nnz,
      const ALPHA_Number *alpha,
      const alpha_dcu_matrix_descr_t descr,
      const ALPHA_Number *csr_val,
      const ALPHA_INT *csr_row_ptr,
      const ALPHA_INT *csr_col_ind,
      ALPHA_Number *B,
      ALPHA_INT ldb,
      alphasparse_dcu_mat_info_t info,
      alphasparse_dcu_solve_policy_t policy,
      void *temp_buffer)
{
    // Check for valid handle and matrix descriptor
    if (handle == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
    } else if (descr == nullptr) // || info == nullptr)
    {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Check operation type
    if (trans_A == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE || trans_A == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (trans_B == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE || trans_B == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    // Check index base
    if (descr->base != ALPHA_SPARSE_INDEX_BASE_ZERO && descr->base != ALPHA_SPARSE_INDEX_BASE_ONE) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Check sizes
    if (m < 0 || nrhs < 0 || nnz < 0) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return if possible
    if (m == 0 || nrhs == 0 || nnz == 0) {
        return ALPHA_SPARSE_STATUS_SUCCESS;
    }

    // Check pointer arguments
    if (csr_val == nullptr || csr_row_ptr == nullptr || csr_col_ind == nullptr || alpha == nullptr || B == nullptr) // || temp_buffer == nullptr)
    {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }
    ALPHA_Number l_alpha = *alpha;

    if (descr->type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR) {
        // printf("index: %d\n", index4(trans_B, trans_A, descr->mode, descr->diag,
        //                              ALPHA_SPARSE_OPERATION_NUM,
        //                              ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM));
        check_null_return(trsm_csr_diag_fill_operation[index4(trans_B, trans_A, descr->mode, descr->diag, ALPHA_SPARSE_OPERATION_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)],
                          ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
        return trsm_csr_diag_fill_operation[index4(trans_B, trans_A, descr->mode, descr->diag, ALPHA_SPARSE_OPERATION_NUM, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](handle, m, nrhs, nnz, l_alpha, csr_val, csr_row_ptr, csr_col_ind, B, ldb, info, policy, temp_buffer);
    } else {
        // todo
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
