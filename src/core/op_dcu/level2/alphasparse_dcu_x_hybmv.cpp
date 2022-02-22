#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "alphasparse/kernel_dcu.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      alphasparse_operation_t trans,
      const ALPHA_Number *alpha,
      const alpha_dcu_matrix_descr_t descr,
      const ALPHA_SPMAT_HYB *hyb,
      const ALPHA_Number *x,
      const ALPHA_Number *beta,
      ALPHA_Number *y)
{
    // Check for valid handle and matrix descriptor
    if (handle == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
    }

    if (descr == nullptr || hyb == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Check index base
    if (descr->base != ALPHA_SPARSE_INDEX_BASE_ZERO && descr->base != ALPHA_SPARSE_INDEX_BASE_ONE) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    const ALPHA_INT ell_nnz = hyb->ell_width * hyb->rows;
    const ALPHA_INT coo_nnz = hyb->nnz;

    // Check sizes
    if (hyb->rows < 0 || hyb->cols < 0 || ell_nnz + coo_nnz < 0) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Check ELL-HYB structure
    if (ell_nnz > 0) {
        if (hyb->ell_width < 0) {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        } else if (hyb->d_ell_col_ind == nullptr || hyb->d_ell_val == nullptr) {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }

    // Check COO-HYB structure
    if (coo_nnz > 0) {
        if (hyb->d_coo_row_val == nullptr || hyb->d_coo_col_val == nullptr || hyb->d_coo_val == nullptr) {
            return ALPHA_SPARSE_STATUS_INVALID_POINTER;
        }
    }

    if (x == nullptr || y == nullptr || alpha == nullptr || beta == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Quick return if possible
    if (hyb->rows == 0 || hyb->cols == 0 || ell_nnz + coo_nnz == 0) {
        return ALPHA_SPARSE_STATUS_SUCCESS;
    }

    ALPHA_Number l_alpha = *alpha;
    ALPHA_Number l_beta  = *beta;

    if (descr->type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL) {
        if (ell_nnz > 0)
            dcu_gemv_ell(handle,
                         hyb->rows,
                         hyb->cols,
                         *alpha,
                         (ALPHA_Number *)hyb->d_ell_val,
                         hyb->d_ell_col_ind,
                         hyb->ell_width,
                         x,
                         *beta,
                         y);

        if (coo_nnz > 0) {
            if (ell_nnz > 0) {
                ALPHA_Number one;
#if COMPLEX
                one = {1, 0};
#else
                one = 1;
#endif
                dcu_gemv_coo(handle,
                             hyb->rows,
                             hyb->cols,
                             coo_nnz,
                             *alpha,
                             (ALPHA_Number *)hyb->d_coo_val,
                             hyb->d_coo_row_val,
                             hyb->d_coo_col_val,
                             x,
                             one,
                             y);
            } else {
                dcu_gemv_coo(handle,
                             hyb->rows,
                             hyb->cols,
                             coo_nnz,
                             *alpha,
                             (ALPHA_Number *)hyb->d_coo_val,
                             hyb->d_coo_row_val,
                             hyb->d_coo_col_val,
                             x,
                             *beta,
                             y);
            }
        }

        return ALPHA_SPARSE_STATUS_SUCCESS;
    } else {
        // todo
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
