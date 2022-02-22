#include "alphasparse/handle.h"
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "alphasparse/spmat.h"
#include "alphasparse/kernel_dcu.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

static alphasparse_status_t (*dcu_gemv_csr5_operation[])(alphasparse_dcu_handle_t handle,
                                                         const ALPHA_Number alpha,
                                                         const ALPHA_SPMAT_CSR5 *csr5,
                                                         alphasparse_dcu_mat_info_t info,
                                                         const ALPHA_Number *x,
                                                         const ALPHA_Number beta,
                                                         ALPHA_Number *y) = {
    dcu_gemv_csr5,
};

alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      alphasparse_operation_t trans,
      const ALPHA_Number *alpha,
      const alpha_dcu_matrix_descr_t descr,
      const ALPHA_SPMAT_CSR5 *csr5,
      alphasparse_dcu_mat_info_t info,
      const ALPHA_Number *x,
      const ALPHA_Number *beta,
      ALPHA_Number *y)
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

    // Check sizes
    if (csr5->num_rows < 0 || csr5->num_cols < 0 || csr5->nnz < 0) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return if possible
    if (csr5->num_rows == 0 || csr5->num_cols == 0 || csr5->nnz == 0) {
        return ALPHA_SPARSE_STATUS_SUCCESS;
    }

    // Check pointer arguments
    if (alpha == nullptr || beta == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Check the rest of pointer arguments
    if (!csr5 || !csr5->row_ptr || !csr5->col_idx || !csr5->val || !csr5->tile_ptr || !csr5->tile_desc || !csr5->tile_desc_offset_ptr || !csr5->calibrator) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    ALPHA_Number l_alpha = *alpha;
    ALPHA_Number l_beta  = *beta;

    if (descr->type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL) {
        if (trans == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
            return dcu_gemv_csr5_operation[0](handle, l_alpha, csr5, info, x, l_beta, y);
        } else {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    } else {
        // doto
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
