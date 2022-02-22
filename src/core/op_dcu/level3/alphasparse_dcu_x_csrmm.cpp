#include "alphasparse/handle.h"
#include "alphasparse/spapi_dcu.h"
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

static alphasparse_status_t (*dcu_gemm_csr_operation[])(alphasparse_dcu_handle_t handle,
                                                        ALPHA_INT m,
                                                        ALPHA_INT n,
                                                        ALPHA_INT k,
                                                        ALPHA_INT nnz,
                                                        ALPHA_Number alpha,
                                                        const ALPHA_Number *csr_val,
                                                        const ALPHA_INT *csr_row_ptr,
                                                        const ALPHA_INT *csr_col_ind,
                                                        const ALPHA_Number *B,
                                                        ALPHA_INT ldb,
                                                        ALPHA_Number beta,
                                                        ALPHA_Number *C,
                                                        ALPHA_INT ldc) = {
    dcu_gemm_csr_col,
    dcu_gemm_csr_col_transA,
    dcu_gemm_csr_col_transB,
    dcu_gemm_csr_col_transAB,
    dcu_gemm_csr_row,
    dcu_gemm_csr_row_transA,
    dcu_gemm_csr_row_transB,
    dcu_gemm_csr_row_transAB,
#ifdef COMPLEX
    NULL, // dcu_gemm_csr_col_conjA,
    NULL, // dcu_gemm_csr_col_conjB,
    NULL, // dcu_gemm_csr_col_conjAB,
    NULL, // dcu_gemm_csr_col_transAconjB
    NULL, // dcu_gemm_csr_col_conjAtransB
    NULL, // dcu_gemm_csr_row_conjA,
    NULL, // dcu_gemm_csr_row_conjB,
    NULL, // dcu_gemm_csr_row_conjAB,
    NULL, // dcu_gemm_csr_row_transAconjB
    NULL // dcu_gemm_csr_row_conjAtransB
#endif
};

alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      alphasparse_operation_t trans_A,
      alphasparse_operation_t trans_B,
      alphasparse_layout_t layout,
      ALPHA_INT m,
      ALPHA_INT n,
      ALPHA_INT k,
      ALPHA_INT nnz,
      const ALPHA_Number *alpha,
      const alpha_dcu_matrix_descr_t descr,
      const ALPHA_Number *csr_val,
      const ALPHA_INT *csr_row_ptr,
      const ALPHA_INT *csr_col_ind,
      const ALPHA_Number *B,
      ALPHA_INT ldb,
      const ALPHA_Number *beta,
      ALPHA_Number *matC,
      ALPHA_INT ldc)
{
    // Check for valid handle and matrix descriptor
    alphasparse_operation_t operationA = trans_A;
    alphasparse_operation_t operationB = trans_B;
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
    if (m < 0 || n < 0 || k <= 0 || nnz < 0 || ldc < 0 || ldb < 0) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return if possible
    if (m == 0 || n == 0 || nnz == 0 || ldc == 0 || ldb == 0) {
        return ALPHA_SPARSE_STATUS_SUCCESS;
    }

    // Check pointer arguments
    if (alpha == nullptr || beta == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    //
    // Check the rest of pointer arguments
    //
    if (csr_val == nullptr || csr_row_ptr == nullptr || csr_col_ind == nullptr || B == nullptr || matC == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }
    ALPHA_Number l_alpha = *alpha;
    ALPHA_Number l_beta  = *beta;

    // row_major
    if (layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR) {
        if (descr->type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL) {
            if (operationA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
                if (trans_B == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
                    check_null_return(dcu_gemm_csr_operation[4], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
                    return dcu_gemm_csr_operation[4]( //dcu_gemm_csr
                        handle,
                        m,
                        n,
                        k,
                        nnz,
                        l_alpha,
                        csr_val,
                        csr_row_ptr,
                        csr_col_ind,
                        B,
                        ldb,
                        l_beta,
                        matC,
                        ldc);
                } else if (trans_B == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
                    // todo
                    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                } else {
                    // todo
                    // dcu_gemm_csr_conjB
                    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
            } else {
                // todo
                // dcu_gemm_csr_transA,
                // dcu_gemm_csr_transAB,
                // dcu_gemm_csr_conjA,
                // dcu_gemm_csr_conjAB,
                // dcu_gemm_csr_transAconjB
                // dcu_gemm_csr_transAconjB
                // dcu_gemm_csr_congjAtransB
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
        } else {
            // todo
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    }

    // col_major
    if (layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR) {
        if (descr->type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL) {
            if (operationA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
                if (trans_B == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
                    check_null_return(dcu_gemm_csr_operation[0], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
                    return dcu_gemm_csr_operation[0]( //dcu_gemm_csr
                        handle,
                        m,
                        n,
                        k,
                        nnz,
                        l_alpha,
                        csr_val,
                        csr_row_ptr,
                        csr_col_ind,
                        B,
                        ldb,
                        l_beta,
                        matC,
                        ldc);
                } else if (trans_B == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
                    check_null_return(dcu_gemm_csr_operation[2], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
                    return dcu_gemm_csr_operation[2]( //dcu_gemm_csr_transB
                        handle,
                        m,
                        n,
                        k,
                        nnz,
                        l_alpha,
                        csr_val,
                        csr_row_ptr,
                        csr_col_ind,
                        B,
                        ldb,
                        l_beta,
                        matC,
                        ldc);
                } else {
                    // todo
                    // dcu_gemm_csr_conjB
                    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
            } else {
                // todo
                // dcu_gemm_csr_transA,
                // dcu_gemm_csr_transAB,
                // dcu_gemm_csr_conjA,
                // dcu_gemm_csr_conjAB,
                // dcu_gemm_csr_transAconjB
                // dcu_gemm_csr_transAconjB
                // dcu_gemm_csr_congjAtransB
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
        } else {
            // todo
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
