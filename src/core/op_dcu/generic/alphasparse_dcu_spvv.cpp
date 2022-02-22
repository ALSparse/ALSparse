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
alphasparse_dcu_spvv(alphasparse_dcu_handle_t handle,
                     alphasparse_operation_t trans,
                     const alphasparse_dcu_spvec_descr_t x,
                     const alphasparse_dcu_dnvec_descr_t y,
                     void *result,
                     alphasparse_datatype_t compute_type,
                     size_t *buffer_size,
                     void *temp_buffer)
{
    // Check for valid handle and matrix descriptor
    if (handle == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
    }

    if (buffer_size == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // If temp_buffer is nullptr, return buffer_size
    if (temp_buffer == nullptr) {
        // We do not need a buffer
        *buffer_size = 4;

        return ALPHA_SPARSE_STATUS_SUCCESS;
    }

    //
    // Check the rest of pointer arguments
    //
    if (x == nullptr || y == nullptr || result == nullptr || temp_buffer == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    if (x->init == false || y->init == false)
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;

    if (x->data_type != y->data_type || x->data_type != compute_type) {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    // real precision
    if (x->idx_type == ALPHA_SPARSE_DCU_INDEXTYPE_I32 && compute_type == ALPHA_SPARSE_DATATYPE_FLOAT) {
        dcu_s_doti(handle, x->nnz, (float *)x->val_data, (ALPHA_INT *)x->idx_data, (float *)y->values, (float *)result);
    }

    if (x->idx_type == ALPHA_SPARSE_DCU_INDEXTYPE_I32 && compute_type == ALPHA_SPARSE_DATATYPE_DOUBLE) {
        dcu_d_doti(handle, x->nnz, (double *)x->val_data, (ALPHA_INT *)x->idx_data, (double *)y->values, (double *)result);
    }

    // complex precision
    if (x->idx_type == ALPHA_SPARSE_DCU_INDEXTYPE_I32 && compute_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
        // non transpose
        if (trans == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
            dcu_c_doti(handle, x->nnz, (ALPHA_Complex8 *)x->val_data, (ALPHA_INT *)x->idx_data, (ALPHA_Complex8 *)y->values, (ALPHA_Complex8 *)result);
        }

        // conjugate transpose
        if (trans == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
            dcu_c_dotci(handle, x->nnz, (ALPHA_Complex8 *)x->val_data, (ALPHA_INT *)x->idx_data, (ALPHA_Complex8 *)y->values, (ALPHA_Complex8 *)result);
        }
    }

    if (x->idx_type == ALPHA_SPARSE_DCU_INDEXTYPE_I32 && compute_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
        // non transpose
        if (trans == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
            dcu_z_doti(handle, x->nnz, (ALPHA_Complex16 *)x->val_data, (ALPHA_INT *)x->idx_data, (ALPHA_Complex16 *)y->values, (ALPHA_Complex16 *)result);
        }

        // conjugate transpose
        if (trans == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
            dcu_z_dotci(handle, x->nnz, (ALPHA_Complex16 *)x->val_data, (ALPHA_INT *)x->idx_data, (ALPHA_Complex16 *)y->values, (ALPHA_Complex16 *)result);
        }
    }

    //TODO support ALPHA_SPARSE_DCU_INDEXTYPE_I64

    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
