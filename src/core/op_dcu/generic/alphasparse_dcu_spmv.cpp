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
#include "alphasparse/util/error.h"

alphasparse_status_t
alphasparse_dcu_spmv(alphasparse_dcu_handle_t handle,
                     alphasparse_operation_t trans,
                     const void *alpha,
                     const alphasparse_dcu_spmat_descr_t mat,
                     const alphasparse_dcu_dnvec_descr_t x,
                     const void *beta,
                     const alphasparse_dcu_dnvec_descr_t y,
                     alphasparse_datatype_t compute_type,
                     alphasparse_dcu_spmv_alg_t alg,
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
    if (x == nullptr || y == nullptr || mat == nullptr || alpha == nullptr || beta == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Check if descriptors are initialized
    if (mat->init == false || x->init == false || y->init == false) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Check for matching types while we do not support mixed precision computation
    if (compute_type != mat->data_type || compute_type != x->data_type || compute_type != y->data_type) {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    // TODO
    if (mat->row_type != mat->col_type || mat->row_type != ALPHA_SPARSE_DCU_INDEXTYPE_I32) {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    // If temp_buffer is nullptr, return buffer_size
    if (temp_buffer == nullptr) {
        // We do not need a buffer
        *buffer_size = 4;

        //TODO analysis did nothing current
        // // Run CSR analysis step when format is CSR
        if (mat->format == ALPHA_SPARSE_FORMAT_CSR) {
            // If algorithm 1 or default is selected and analysis step is required
            if ((alg == ALPHA_SPARSE_DCU_SPMV_ALG_DEFAULT || alg == ALPHA_SPARSE_DCU_SPMV_ALG_CSR_ADAPTIVE) && mat->analysed == false) {
                // Analyse CSR matrix
                alpha_call_exit(alphasparse_dcu_s_csrmv_analysis(
                                    handle, trans, mat->rows, mat->cols, mat->nnz, mat->descr, (float *)mat->val_data, (ALPHA_INT *)mat->row_data, (ALPHA_INT *)mat->col_data, mat->info),
                                "alphasparse_s_analysis_csrmv");

                mat->analysed = true;
            }
        }

        return ALPHA_SPARSE_STATUS_SUCCESS;
    }

    // COO
    if (mat->format == ALPHA_SPARSE_FORMAT_COO) {
        if (mat->data_type == ALPHA_SPARSE_DATATYPE_FLOAT)
            return alphasparse_dcu_s_coomv(handle,
                                           trans,
                                           mat->rows,
                                           mat->cols,
                                           mat->nnz,
                                           (float *)alpha,
                                           mat->descr,
                                           (float *)mat->val_data,
                                           (ALPHA_INT *)mat->row_data,
                                           (ALPHA_INT *)mat->col_data,
                                           (float *)x->values,
                                           (float *)beta,
                                           (float *)y->values);

        if (mat->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE)
            return alphasparse_dcu_d_coomv(handle,
                                           trans,
                                           mat->rows,
                                           mat->cols,
                                           mat->nnz,
                                           (double *)alpha,
                                           mat->descr,
                                           (double *)mat->val_data,
                                           (ALPHA_INT *)mat->row_data,
                                           (ALPHA_INT *)mat->col_data,
                                           (double *)x->values,
                                           (double *)beta,
                                           (double *)y->values);

        if (mat->data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
            return alphasparse_dcu_c_coomv(handle,
                                           trans,
                                           mat->rows,
                                           mat->cols,
                                           mat->nnz,
                                           (ALPHA_Complex8 *)alpha,
                                           mat->descr,
                                           (ALPHA_Complex8 *)mat->val_data,
                                           (ALPHA_INT *)mat->row_data,
                                           (ALPHA_INT *)mat->col_data,
                                           (ALPHA_Complex8 *)x->values,
                                           (ALPHA_Complex8 *)beta,
                                           (ALPHA_Complex8 *)y->values);

        if (mat->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
            return alphasparse_dcu_z_coomv(handle,
                                           trans,
                                           mat->rows,
                                           mat->cols,
                                           mat->nnz,
                                           (ALPHA_Complex16 *)alpha,
                                           mat->descr,
                                           (ALPHA_Complex16 *)mat->val_data,
                                           (ALPHA_INT *)mat->row_data,
                                           (ALPHA_INT *)mat->col_data,
                                           (ALPHA_Complex16 *)x->values,
                                           (ALPHA_Complex16 *)beta,
                                           (ALPHA_Complex16 *)y->values);
    }

    // COO (AoS)
    if (mat->format == ALPHA_SPARSE_FORMAT_COO_AOS) {
        // todo, support coo aos format
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    // CSR
    if (mat->format == ALPHA_SPARSE_FORMAT_CSR) {
        if (mat->data_type == ALPHA_SPARSE_DATATYPE_FLOAT)
            alphasparse_dcu_s_csrmv(handle,
                                    trans,
                                    mat->rows,
                                    mat->cols,
                                    mat->nnz,
                                    (float *)alpha,
                                    mat->descr,
                                    (float *)mat->val_data,
                                    (ALPHA_INT *)mat->row_data,
                                    (ALPHA_INT *)mat->col_data,
                                    mat->info,
                                    (float *)x->values,
                                    (float *)beta,
                                    (float *)y->values);

        if (mat->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE)
            alphasparse_dcu_d_csrmv(handle,
                                    trans,
                                    mat->rows,
                                    mat->cols,
                                    mat->nnz,
                                    (double *)alpha,
                                    mat->descr,
                                    (double *)mat->val_data,
                                    (ALPHA_INT *)mat->row_data,
                                    (ALPHA_INT *)mat->col_data,
                                    mat->info,
                                    (double *)x->values,
                                    (double *)beta,
                                    (double *)y->values);

        if (mat->data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
            alphasparse_dcu_c_csrmv(handle,
                                    trans,
                                    mat->rows,
                                    mat->cols,
                                    mat->nnz,
                                    (ALPHA_Complex8 *)alpha,
                                    mat->descr,
                                    (ALPHA_Complex8 *)mat->val_data,
                                    (ALPHA_INT *)mat->row_data,
                                    (ALPHA_INT *)mat->col_data,
                                    mat->info,
                                    (ALPHA_Complex8 *)x->values,
                                    (ALPHA_Complex8 *)beta,
                                    (ALPHA_Complex8 *)y->values);

        if (mat->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
            alphasparse_dcu_z_csrmv(handle,
                                    trans,
                                    mat->rows,
                                    mat->cols,
                                    mat->nnz,
                                    (ALPHA_Complex16 *)alpha,
                                    mat->descr,
                                    (ALPHA_Complex16 *)mat->val_data,
                                    (ALPHA_INT *)mat->row_data,
                                    (ALPHA_INT *)mat->col_data,
                                    mat->info,
                                    (ALPHA_Complex16 *)x->values,
                                    (ALPHA_Complex16 *)beta,
                                    (ALPHA_Complex16 *)y->values);
    }

    // ELL
    if (mat->format == ALPHA_SPARSE_FORMAT_ELL) {
        if (mat->data_type == ALPHA_SPARSE_DATATYPE_FLOAT)
            alphasparse_dcu_s_ellmv(handle,
                                    trans,
                                    mat->rows,
                                    mat->cols,
                                    (float *)alpha,
                                    mat->descr,
                                    (float *)mat->val_data,
                                    (ALPHA_INT *)mat->col_data,
                                    mat->nnz, //?
                                    (float *)x->values,
                                    (float *)beta,
                                    (float *)y->values);

        if (mat->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE)
            alphasparse_dcu_d_ellmv(handle,
                                    trans,
                                    mat->rows,
                                    mat->cols,
                                    (double *)alpha,
                                    mat->descr,
                                    (double *)mat->val_data,
                                    (ALPHA_INT *)mat->col_data,
                                    mat->nnz, //?
                                    (double *)x->values,
                                    (double *)beta,
                                    (double *)y->values);

        if (mat->data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
            alphasparse_dcu_c_ellmv(handle,
                                    trans,
                                    mat->rows,
                                    mat->cols,
                                    (ALPHA_Complex8 *)alpha,
                                    mat->descr,
                                    (ALPHA_Complex8 *)mat->val_data,
                                    (ALPHA_INT *)mat->col_data,
                                    mat->nnz, //?
                                    (ALPHA_Complex8 *)x->values,
                                    (ALPHA_Complex8 *)beta,
                                    (ALPHA_Complex8 *)y->values);

        if (mat->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
            alphasparse_dcu_z_ellmv(handle,
                                    trans,
                                    mat->rows,
                                    mat->cols,
                                    (ALPHA_Complex16 *)alpha,
                                    mat->descr,
                                    (ALPHA_Complex16 *)mat->val_data,
                                    (ALPHA_INT *)mat->col_data,
                                    mat->nnz, //?
                                    (ALPHA_Complex16 *)x->values,
                                    (ALPHA_Complex16 *)beta,
                                    (ALPHA_Complex16 *)y->values);
    }

    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
