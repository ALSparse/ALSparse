#include "alphasparse/handle.h"
#include "alphasparse/spapi_dcu.h"
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "alphasparse/spapi_dcu.h"
#include "alphasparse/kernel_dcu.h"
#include "alphasparse/util_dcu.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/util/check.h"

// static alphasparse_status_t (*dcu_geam_nnz_csr[])(alphasparse_dcu_handle_t handle,
//                                                  ALPHA_INT m,
//                                                  ALPHA_INT n,
//                                                  ALPHA_INT nnz_A,
//                                                  const ALPHA_INT *csr_row_ptr_A,
//                                                  const ALPHA_INT *csr_col_ind_A,
//                                                  ALPHA_INT nnz_B,
//                                                  const ALPHA_INT *csr_row_ptr_B,
//                                                  const ALPHA_INT *csr_col_ind_B,
//                                                  ALPHA_INT *csr_row_ptr_C,
//                                                  ALPHA_INT *nnz_C) = {
//     dcu_geam_nnz_csr,
// };

alphasparse_status_t
alphasparse_dcu_csrgeam_nnz(alphasparse_dcu_handle_t handle,
                            ALPHA_INT m,
                            ALPHA_INT n,
                            const alpha_dcu_matrix_descr_t descr_A,
                            ALPHA_INT nnz_A,
                            const ALPHA_INT *csr_row_ptr_A,
                            const ALPHA_INT *csr_col_ind_A,
                            const alpha_dcu_matrix_descr_t descr_B,
                            ALPHA_INT nnz_B,
                            const ALPHA_INT *csr_row_ptr_B,
                            const ALPHA_INT *csr_col_ind_B,
                            const alpha_dcu_matrix_descr_t descr_C,
                            ALPHA_INT *csr_row_ptr_C,
                            ALPHA_INT *nnz_C)
{
    // Check for valid handle and matrix descriptor
    if (handle == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
    } else if (descr_A == nullptr || descr_B == nullptr || descr_C == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Check index base
    if (descr_A->base != ALPHA_SPARSE_INDEX_BASE_ZERO && descr_A->base != ALPHA_SPARSE_INDEX_BASE_ONE) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    if (descr_B->base != ALPHA_SPARSE_INDEX_BASE_ZERO && descr_B->base != ALPHA_SPARSE_INDEX_BASE_ONE) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    if (descr_C->base != ALPHA_SPARSE_INDEX_BASE_ZERO && descr_C->base != ALPHA_SPARSE_INDEX_BASE_ONE) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Check sizes
    if (m < 0 || n < 0 || nnz_A < 0 || nnz_B < 0) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Check for valid nnz_C pointer
    if (nnz_C == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Quick return if possible
    if (m == 0 || n == 0 || nnz_A == 0 || nnz_B == 0) {
        *nnz_C = 0;
        return ALPHA_SPARSE_STATUS_SUCCESS;
    }

    // Check valid pointers
    if (csr_row_ptr_A == nullptr || csr_col_ind_A == nullptr || csr_row_ptr_B == nullptr || csr_col_ind_B == nullptr || csr_row_ptr_C == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    if (descr_A->type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL) {
        if (descr_B->type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL) {
            if (descr_C->type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL) {
                check_null_return(dcu_geam_nnz_csr,
                                  ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
                return dcu_geam_nnz_csr(handle, m, n, nnz_A, csr_row_ptr_A, csr_col_ind_A, nnz_B, csr_row_ptr_B, csr_col_ind_B, csr_row_ptr_C, nnz_C);
            } else {
                // todo
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
        } else {
            // todo
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    } else {
        // todo
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
