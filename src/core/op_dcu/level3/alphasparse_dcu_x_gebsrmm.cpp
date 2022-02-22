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

static alphasparse_status_t (*dcu_gemm_gebsr_operation[])(alphasparse_dcu_handle_t handle,
                                                          alphasparse_layout_t dir,
                                                          ALPHA_INT mb,
                                                          ALPHA_INT n,
                                                          ALPHA_INT kb,
                                                          ALPHA_INT nnzb,
                                                          const ALPHA_Number alpha,
                                                          const ALPHA_Number *bsr_val,
                                                          const ALPHA_INT *bsr_row_ptr,
                                                          const ALPHA_INT *bsr_col_ind,
                                                          ALPHA_INT row_block_dim,
                                                          ALPHA_INT col_block_dim,
                                                          const ALPHA_Number *B,
                                                          ALPHA_INT ldb,
                                                          const ALPHA_Number beta,
                                                          ALPHA_Number *C,
                                                          ALPHA_INT ldc) = {
    dcu_gemm_gebsr,
    NULL, //dcu_gemm_gebsr_transA,
    dcu_gemm_gebsr_transB,
    dcu_gemm_gebsr_transAB,
#ifdef COMPLEX
    NULL, // dcu_gemm_gebsr_conjA,
    NULL, // dcu_gemm_gebsr_conjB,
    NULL, // dcu_gemm_gebsr_conjAB,
    NULL, // dcu_gemm_gebsr_transAconjB
    NULL // dcu_gemm_gebsr_congjAtransB
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
      ALPHA_INT row_block_dim,
      ALPHA_INT col_block_dim,
      const ALPHA_Number *matB,
      ALPHA_INT ldb,
      const ALPHA_Number *beta,
      ALPHA_Number *matC,
      ALPHA_INT ldc)
{
    // Check for valid handle and matrix descriptor
    // Check for valid handle and matrix descriptor
    if (handle == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
    } else if (descr == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Check index base
    if (descr->base != ALPHA_SPARSE_INDEX_BASE_ZERO && descr->base != ALPHA_SPARSE_INDEX_BASE_ONE) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Check sizes
    if (mb < 0 || n < 0 || kb < 0 || nnzb < 0 || row_block_dim <= 0 || col_block_dim <= 0) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return if possible
    if (mb == 0 || n == 0 || kb == 0) {
        return ALPHA_SPARSE_STATUS_SUCCESS;
    }

    // Check pointer arguments
    if (bsr_val == nullptr || bsr_row_ptr == nullptr || bsr_col_ind == nullptr || matB == nullptr || matC == nullptr || alpha == nullptr || beta == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Check leading dimension of B
    if (trans_B == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
        if (ldb < kb) {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    } else {
        if (ldb < n) {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }

    // Check leading dimension of C
    if (ldc < mb) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    ALPHA_Number l_alpha = *alpha;
    ALPHA_Number l_beta  = *beta;

    if (descr->type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL) {
        if (trans_A == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
            if (trans_B == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
                check_null_return(dcu_gemm_gebsr_operation[0], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
                return dcu_gemm_gebsr_operation[0]( //dcu_gemm_bsr
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
                    row_block_dim,
                    col_block_dim,
                    matB,
                    ldb,
                    l_beta,
                    matC,
                    ldc);
            } else if (trans_B == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
                check_null_return(dcu_gemm_gebsr_operation[2], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
                return dcu_gemm_gebsr_operation[2]( //dcu_gemm_bsr_transB
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
                    row_block_dim,
                    col_block_dim,
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
