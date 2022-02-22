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

static alphasparse_status_t (*dcu_gemm_bsr_operation[])(alphasparse_dcu_handle_t handle,
                                                        alphasparse_layout_t dir,
                                                        ALPHA_INT mb,
                                                        ALPHA_INT n,
                                                        ALPHA_INT kb,
                                                        ALPHA_INT nnzb,
                                                        const ALPHA_Number alpha,
                                                        const ALPHA_Number *bsr_val,
                                                        const ALPHA_INT *bsr_row_ptr,
                                                        const ALPHA_INT *bsr_col_ind,
                                                        ALPHA_INT block_dim,
                                                        const ALPHA_Number *B,
                                                        ALPHA_INT ldb,
                                                        const ALPHA_Number beta,
                                                        ALPHA_Number *C,
                                                        ALPHA_INT ldc) = {
    dcu_gemm_bsr,
    dcu_gemm_bsr_transA,
    dcu_gemm_bsr_transB,
    dcu_gemm_bsr_transAB,
#ifdef COMPLEX
    NULL, // dcu_gemm_bsr_conjA,
    NULL, // dcu_gemm_bsr_conjB,
    NULL, // dcu_gemm_bsr_conjAB,
    NULL, // dcu_gemm_bsr_transAconjB
    NULL // dcu_gemm_bsr_congjAtransB
#endif
};

alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      alphasparse_layout_t dir,
      alphasparse_operation_t trans_A,
      alphasparse_operation_t trans_B,
      ALPHA_INT mb,
      ALPHA_INT n,
      ALPHA_INT kb,
      ALPHA_INT nnzb,
      const ALPHA_Number *alpha,
      const alpha_dcu_matrix_descr_t descr,
      const ALPHA_Number *bsr_val,
      const ALPHA_INT *bsr_row_ptr,
      const ALPHA_INT *bsr_col_ind,
      ALPHA_INT block_dim,
      const ALPHA_Number *matB,
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
    if (mb < 0 || n < 0 || kb <= 0 || nnzb < 0 || ldc < 0 || ldb < 0 || block_dim <= 0) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return if possible
    if (mb == 0 || n == 0 || kb == 0 || nnzb == 0 || ldc == 0 || ldb == 0) {
        return ALPHA_SPARSE_STATUS_SUCCESS;
    }

    // Check pointer arguments
    if (alpha == nullptr || beta == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    //
    // Check the rest of pointer arguments
    //
    if (bsr_val == nullptr || bsr_row_ptr == nullptr || bsr_col_ind == nullptr || matB == nullptr || matC == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }
    ALPHA_Number l_alpha = *alpha;
    ALPHA_Number l_beta  = *beta;

    if (descr->type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL) {
        if (operationA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
            if (trans_B == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
                check_null_return(dcu_gemm_bsr_operation[0], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
                return dcu_gemm_bsr_operation[0]( //dcu_gemm_bsr
                    handle,
                    dir,
                    mb,
                    n,
                    kb,
                    nnzb,
                    l_alpha,
                    bsr_val,
                    bsr_row_ptr,
                    bsr_col_ind,
                    block_dim,
                    matB,
                    ldb,
                    l_beta,
                    matC,
                    ldc);
            } else if (trans_B == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
                check_null_return(dcu_gemm_bsr_operation[2], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
                return dcu_gemm_bsr_operation[2]( //dcu_gemm_bsr_transB
                    handle,
                    dir,
                    mb,
                    n,
                    kb,
                    nnzb,
                    l_alpha,
                    bsr_val,
                    bsr_row_ptr,
                    bsr_col_ind,
                    block_dim,
                    matB,
                    ldb,
                    l_beta,
                    matC,
                    ldc);
            } else {
                // todo
                // dcu_gemm_bsr_conjB
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
        } else {
            // todo
            // dcu_gemm_bsr_transA,
            // dcu_gemm_bsr_transAB,
            // dcu_gemm_bsr_conjA,
            // dcu_gemm_bsr_conjAB,
            // dcu_gemm_bsr_transAconjB
            // dcu_gemm_bsr_transAconjB
            // dcu_gemm_bsr_congjAtransB
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
