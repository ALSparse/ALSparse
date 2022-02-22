#include "alphasparse/handle.h"
#include "alphasparse/spapi_dcu.h"
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "alphasparse/spapi_dcu.h"
#include "alphasparse/kernel_dcu.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/util/check.h"

alphasparse_status_t
alphasparse_dcu_spgemm(alphasparse_dcu_handle_t handle,
                       alphasparse_operation_t trans_A,
                       alphasparse_operation_t trans_B,
                       const void *alpha,
                       const alphasparse_dcu_spmat_descr_t matA,
                       const alphasparse_dcu_spmat_descr_t matB,
                       const void *beta,
                       const alphasparse_dcu_spmat_descr_t matD,
                       alphasparse_dcu_spmat_descr_t matC,
                       alphasparse_datatype_t compute_type,
                       alphasparse_dcu_spgemm_alg_t alg,
                       alphasparse_dcu_spgemm_stage_t stage,
                       size_t *buffer_size,
                       void *temp_buffer)
{
    // Check for valid handle and matrix descriptor
    if (handle == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
    }

    //
    // Check the rest of pointer arguments
    //
    if (matA == nullptr || matB == nullptr || matC == nullptr || matD == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Check for valid scalars
    if (alpha == nullptr && beta == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Check for valid buffer_size pointer only if temp_buffer is nullptr
    if (temp_buffer == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Check if descriptors are initialized
    if (matA->init == false || matB->init == false || matC->init == false || matD->init == false) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Check if all sparse matrices are in the same format
    if (matA->format != matB->format || matA->format != matC->format || matA->format != matD->format) {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    // Check for matching data types while we do not support mixed precision computation
    if (compute_type != matA->data_type || compute_type != matB->data_type || compute_type != matC->data_type || compute_type != matD->data_type) {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    // Check for matching index types
    if (matA->row_type != matB->row_type || matA->row_type != matC->row_type || matA->row_type != matD->row_type || matA->col_type != matB->col_type || matA->col_type != matC->col_type || matA->col_type != matD->col_type) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    if (matA->row_type != matA->col_type || matA->row_type != ALPHA_SPARSE_DCU_INDEXTYPE_I32) {
        // TODO
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    if (matA->format != ALPHA_SPARSE_FORMAT_CSR) {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    // step one
    return alphasparse_dcu_csrgemm_nnz(handle,
                                       trans_A,
                                       trans_B,
                                       matA->rows,
                                       matB->cols,
                                       matA->cols,
                                       matA->descr,
                                       matA->nnz,
                                       (ALPHA_INT *)matA->row_data,
                                       (ALPHA_INT *)matA->col_data,
                                       matB->descr,
                                       matB->nnz,
                                       (ALPHA_INT *)matB->row_data,
                                       (ALPHA_INT *)matB->col_data,
                                       matD->descr,
                                       matD->nnz,
                                       (ALPHA_INT *)matD->row_data,
                                       (ALPHA_INT *)matD->col_data,
                                       matC->descr,
                                       (ALPHA_INT *)matC->row_data,
                                       (ALPHA_INT *)&matC->nnz,
                                       matC->info,
                                       temp_buffer);

    // step2
    if (matA->data_type == ALPHA_SPARSE_DATATYPE_FLOAT) {
        alphasparse_dcu_s_csrgemm(handle,
                                  trans_A,
                                  trans_B,
                                  matA->rows,
                                  matB->cols,
                                  matA->cols,
                                  (float *)alpha,
                                  matA->descr,
                                  matA->nnz,
                                  (float *)matA->val_data,
                                  (ALPHA_INT *)matA->row_data,
                                  (ALPHA_INT *)matA->col_data,
                                  matB->descr,
                                  matB->nnz,
                                  (float *)matB->val_data,
                                  (ALPHA_INT *)matB->row_data,
                                  (ALPHA_INT *)matB->col_data,
                                  (float *)beta,
                                  matD->descr,
                                  matD->nnz,
                                  (float *)matD->val_data,
                                  (ALPHA_INT *)matD->row_data,
                                  (ALPHA_INT *)matD->col_data,
                                  matC->descr,
                                  (float *)matC->val_data,
                                  (ALPHA_INT *)matC->row_data,
                                  (ALPHA_INT *)matC->col_data,
                                  matC->info,
                                  temp_buffer);
    }

    if (matA->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE)
        alphasparse_dcu_d_csrgemm(handle,
                                  trans_A,
                                  trans_B,
                                  matA->rows,
                                  matB->cols,
                                  matA->cols,
                                  (double *)alpha,
                                  matA->descr,
                                  matA->nnz,
                                  (double *)matA->val_data,
                                  (ALPHA_INT *)matA->row_data,
                                  (ALPHA_INT *)matA->col_data,
                                  matB->descr,
                                  matB->nnz,
                                  (double *)matB->val_data,
                                  (ALPHA_INT *)matB->row_data,
                                  (ALPHA_INT *)matB->col_data,
                                  (double *)beta,
                                  matD->descr,
                                  matD->nnz,
                                  (double *)matD->val_data,
                                  (ALPHA_INT *)matD->row_data,
                                  (ALPHA_INT *)matD->col_data,
                                  matC->descr,
                                  (double *)matC->val_data,
                                  (ALPHA_INT *)matC->row_data,
                                  (ALPHA_INT *)matC->col_data,
                                  matC->info,
                                  temp_buffer);

    if (matA->data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
        alphasparse_dcu_c_csrgemm(handle,
                                  trans_A,
                                  trans_B,
                                  matA->rows,
                                  matB->cols,
                                  matA->cols,
                                  (ALPHA_Complex8 *)alpha,
                                  matA->descr,
                                  matA->nnz,
                                  (ALPHA_Complex8 *)matA->val_data,
                                  (ALPHA_INT *)matA->row_data,
                                  (ALPHA_INT *)matA->col_data,
                                  matB->descr,
                                  matB->nnz,
                                  (ALPHA_Complex8 *)matB->val_data,
                                  (ALPHA_INT *)matB->row_data,
                                  (ALPHA_INT *)matB->col_data,
                                  (ALPHA_Complex8 *)beta,
                                  matD->descr,
                                  matD->nnz,
                                  (ALPHA_Complex8 *)matD->val_data,
                                  (ALPHA_INT *)matD->row_data,
                                  (ALPHA_INT *)matD->col_data,
                                  matC->descr,
                                  (ALPHA_Complex8 *)matC->val_data,
                                  (ALPHA_INT *)matC->row_data,
                                  (ALPHA_INT *)matC->col_data,
                                  matC->info,
                                  temp_buffer);

    if (matA->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
        alphasparse_dcu_z_csrgemm(handle,
                                  trans_A,
                                  trans_B,
                                  matA->rows,
                                  matB->cols,
                                  matA->cols,
                                  (ALPHA_Complex16 *)alpha,
                                  matA->descr,
                                  matA->nnz,
                                  (ALPHA_Complex16 *)matA->val_data,
                                  (ALPHA_INT *)matA->row_data,
                                  (ALPHA_INT *)matA->col_data,
                                  matB->descr,
                                  matB->nnz,
                                  (ALPHA_Complex16 *)matB->val_data,
                                  (ALPHA_INT *)matB->row_data,
                                  (ALPHA_INT *)matB->col_data,
                                  (ALPHA_Complex16 *)beta,
                                  matD->descr,
                                  matD->nnz,
                                  (ALPHA_Complex16 *)matD->val_data,
                                  (ALPHA_INT *)matD->row_data,
                                  (ALPHA_INT *)matD->col_data,
                                  matC->descr,
                                  (ALPHA_Complex16 *)matC->val_data,
                                  (ALPHA_INT *)matC->row_data,
                                  (ALPHA_INT *)matC->col_data,
                                  matC->info,
                                  temp_buffer);

    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
