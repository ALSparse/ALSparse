#include "alphasparse/handle.h"
#include "alphasparse/spapi_dcu.h"
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

#include "alphasparse/spapi_dcu.h"
#include "alphasparse/kernel_dcu.h"
#include "alphasparse/util_dcu.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/util/check.h"

    static alphasparse_status_t (*spgemm_csr[])(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT n,
                                               ALPHA_INT k,
                                               const ALPHA_Number alpha,
                                               ALPHA_INT nnz_A,
                                               const ALPHA_Number *csr_val_A,
                                               const ALPHA_INT *csr_row_ptr_A,
                                               const ALPHA_INT *csr_col_ind_A,
                                               ALPHA_INT nnz_B,
                                               const ALPHA_Number *csr_val_B,
                                               const ALPHA_INT *csr_row_ptr_B,
                                               const ALPHA_INT *csr_col_ind_B,
                                               const ALPHA_Number beta,
                                               ALPHA_INT nnz_D,
                                               const ALPHA_Number *csr_val_D,
                                               const ALPHA_INT *csr_row_ptr_D,
                                               const ALPHA_INT *csr_col_ind_D,
                                               ALPHA_Number *csr_val_C,
                                               const ALPHA_INT *csr_row_ptr_C,
                                               ALPHA_INT *csr_col_ind_C,
                                               const alphasparse_dcu_mat_info_t info_C,
                                               void *temp_buffer) = {
        dcu_spgemm_csr,
        NULL, //dcu_spgemm_csr_transA,
        NULL, //dcu_spgemm_csr_conjA,
        NULL, //dcu_spgemm_csr_transB,
        NULL, //dcu_spgemm_csr_transAB,
        NULL, //dcu_spgemm_csr_conjAtransB,
        NULL, //dcu_spgemm_csr_conjB,
        NULL, //dcu_spgemm_csr_transAconjB,
        NULL, //dcu_spgemm_csr_conjAB,
    };

    alphasparse_status_t
    ONAME(alphasparse_dcu_handle_t handle,
          alphasparse_operation_t trans_A,
          alphasparse_operation_t trans_B,
          ALPHA_INT m,
          ALPHA_INT n,
          ALPHA_INT k,
          const ALPHA_Number *alpha,
          const alpha_dcu_matrix_descr_t descr_A,
          ALPHA_INT nnz_A,
          const ALPHA_Number *csr_val_A,
          const ALPHA_INT *csr_row_ptr_A,
          const ALPHA_INT *csr_col_ind_A,
          const alpha_dcu_matrix_descr_t descr_B,
          ALPHA_INT nnz_B,
          const ALPHA_Number *csr_val_B,
          const ALPHA_INT *csr_row_ptr_B,
          const ALPHA_INT *csr_col_ind_B,
          const ALPHA_Number *beta,
          const alpha_dcu_matrix_descr_t descr_D,
          ALPHA_INT nnz_D,
          const ALPHA_Number *csr_val_D,
          const ALPHA_INT *csr_row_ptr_D,
          const ALPHA_INT *csr_col_ind_D,
          const alpha_dcu_matrix_descr_t descr_C,
          ALPHA_Number *csr_val_C,
          const ALPHA_INT *csr_row_ptr_C,
          ALPHA_INT *csr_col_ind_C,
          const alphasparse_dcu_mat_info_t info_C,
          void *temp_buffer)
    {
        // Check for valid handle and info structure
        if (handle == nullptr)
        {
            return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
        }
        // else if (info_C == nullptr)
        // {
        //     return rocsparse_status_invalid_pointer;
        // }
        // else if (info_C->csrgemm_info == nullptr)
        // {
        //     return rocsparse_status_internal_error;
        // }

        // Check valid sizes
        if (m < 0 || n < 0 || k < 0 || nnz_A < 0 || nnz_B < 0 || nnz_D < 0)
        {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }

        // Check valid pointers
        if (alpha == nullptr || beta == nullptr || descr_A == nullptr || descr_B == nullptr || descr_C == nullptr || descr_C == nullptr)
        {
            return ALPHA_SPARSE_STATUS_INVALID_POINTER;
        }

        // Check index base
        if (descr_A->base != ALPHA_SPARSE_INDEX_BASE_ZERO && descr_A->base != ALPHA_SPARSE_INDEX_BASE_ONE)
        {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
        if (descr_B->base != ALPHA_SPARSE_INDEX_BASE_ZERO && descr_B->base != ALPHA_SPARSE_INDEX_BASE_ONE)
        {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
        if (descr_D->base != ALPHA_SPARSE_INDEX_BASE_ZERO && descr_D->base != ALPHA_SPARSE_INDEX_BASE_ONE)
        {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }

        // Quick return if possible

        // m == 0 || n == 0 - do nothing
        if (m == 0 || n == 0)
        {
            return ALPHA_SPARSE_STATUS_SUCCESS;
        }

        // todo // k == 0 || nnz_A == 0 || nnz_B == 0 - scale D with beta
        // if (k == 0 || nnz_A == 0 || nnz_B == 0)
        // {
        //     return ALPHA_SPARSE_STATUS_SUCCESS;
        // }

        // Check valid pointers
        if (csr_val_A == nullptr || csr_row_ptr_A == nullptr || csr_col_ind_A == nullptr || csr_val_B == nullptr || csr_row_ptr_B == nullptr || csr_col_ind_B == nullptr || csr_val_D == nullptr || csr_row_ptr_D == nullptr || csr_col_ind_D == nullptr || csr_val_C == nullptr || csr_row_ptr_C == nullptr || csr_col_ind_C == nullptr)
        {
            return ALPHA_SPARSE_STATUS_INVALID_POINTER;
        }
        ALPHA_Number l_alpha = *alpha;
        ALPHA_Number l_beta = *beta;

        if (descr_A->type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
        {
            if (descr_B->type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
            {
                if (descr_C->type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
                {
                    check_null_return(spgemm_csr[index2(trans_B, trans_A, ALPHA_SPARSE_OPERATION_NUM)],
                                      ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
                    return spgemm_csr[index2(trans_B, trans_A,
                                             ALPHA_SPARSE_OPERATION_NUM)](handle, m, n, k, l_alpha,
                                                                        nnz_A, csr_val_A, csr_row_ptr_A, csr_col_ind_A,
                                                                        nnz_B, csr_val_B, csr_row_ptr_B, csr_col_ind_B, l_beta,
                                                                        nnz_D, csr_val_D, csr_row_ptr_D, csr_col_ind_D,
                                                                        csr_val_C, csr_row_ptr_C, csr_col_ind_C,
                                                                        info_C, temp_buffer);
                }
                else
                {
                    // todo
                    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
            }
            else
            {
                // todo
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
        }
        else
        {
            // todo
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }

        return ALPHA_SPARSE_STATUS_SUCCESS;
    }

#ifdef __cplusplus
}
#endif /*__cplusplus */
