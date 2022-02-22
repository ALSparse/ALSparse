#include "alphasparse/handle.h"
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "alphasparse/spapi_dcu.h"
#include "alphasparse/kernel_dcu.h"
#include "alphasparse/util/error.h"
#include "alphasparse/compute.h"
#include "alphasparse/opt.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/util/check.h"

static alphasparse_status_t (*dcu_gemv_csr_operation[])(alphasparse_dcu_handle_t handle,
                                                        ALPHA_INT m,
                                                        ALPHA_INT n,
                                                        ALPHA_INT nnz,
                                                        const ALPHA_Number alpha,
                                                        const ALPHA_Number *csr_val,
                                                        const ALPHA_INT *csr_row_ptr,
                                                        const ALPHA_INT *csr_col_ind,
                                                        alphasparse_dcu_mat_info_t info,
                                                        const ALPHA_Number *x,
                                                        const ALPHA_Number beta,
                                                        ALPHA_Number *y) = {
    dcu_gemv_csr,
    dcu_gemv_csr_trans,
#ifdef COMPLEX
    dcu_gemv_csr_conj,
#endif
};

alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      alphasparse_operation_t trans,
      ALPHA_INT m,
      ALPHA_INT n,
      ALPHA_INT nnz,
      const ALPHA_Number *alpha,
      const alpha_dcu_matrix_descr_t descr,
      const ALPHA_Number *csr_val,
      const ALPHA_INT *csr_row_ptr,
      const ALPHA_INT *csr_col_ind,
      alphasparse_dcu_mat_info_t info,
      const ALPHA_Number *x,
      const ALPHA_Number *beta,
      ALPHA_Number *y)
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
    if (m < 0 || n < 0 || nnz < 0) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return if possible
    if (m == 0 || n == 0 || nnz == 0) {
        return ALPHA_SPARSE_STATUS_SUCCESS;
    }

    // Check pointer arguments
    if (alpha == nullptr || beta == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    //
    // Check the rest of pointer arguments
    //
    if (csr_val == nullptr || csr_row_ptr == nullptr || csr_col_ind == nullptr || x == nullptr || y == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }
    ALPHA_Number l_alpha = *alpha;
    ALPHA_Number l_beta  = *beta;

    if (descr->type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL) {
        check_null_return(dcu_gemv_csr_operation[operation], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
        return dcu_gemv_csr_operation[operation](handle, m, n, nnz, l_alpha, csr_val, csr_row_ptr, csr_col_ind, info, x, l_beta, y);
    } else {
        // doto
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
