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
alphasparse_dcu_gather(alphasparse_dcu_handle_t handle,
                       const alphasparse_dcu_dnvec_descr_t y,
                       alphasparse_dcu_spvec_descr_t x)
{
    // Check for valid handle and matrix descriptor
    if (handle == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
    }

    //
    // Check the rest of pointer arguments
    //
    if (x == nullptr || y == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Check if descriptors are initialized
    if (x->init == false || y->init == false) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Check for matching types while we do not support mixed precision computation
    if (x->data_type != y->data_type) {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    // single real ; i32
    if (x->idx_type == ALPHA_SPARSE_DCU_INDEXTYPE_I32 && x->data_type == ALPHA_SPARSE_DATATYPE_FLOAT) {
        return dcu_s_gthr(handle, x->nnz, (float *)y->values, (float *)x->val_data, (ALPHA_INT *)x->idx_data);
    }
    // double real ; i32
    if (x->idx_type == ALPHA_SPARSE_DCU_INDEXTYPE_I32 && x->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE) {
        return dcu_d_gthr(handle, x->nnz, (double *)y->values, (double *)x->val_data, (ALPHA_INT *)x->idx_data);
    }
    // single complex ; i32
    if (x->idx_type == ALPHA_SPARSE_DCU_INDEXTYPE_I32 && x->data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
        return dcu_c_gthr(handle, x->nnz, (ALPHA_Complex8 *)y->values, (ALPHA_Complex8 *)x->val_data, (ALPHA_INT *)x->idx_data);
    }
    // double complex ; i32
    if (x->idx_type == ALPHA_SPARSE_DCU_INDEXTYPE_I32 && x->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
        return dcu_z_gthr(handle, x->nnz, (ALPHA_Complex16 *)y->values, (ALPHA_Complex16 *)x->val_data, (ALPHA_INT *)x->idx_data);
    }

    //TODO add support to i64
    // // single real ; i64
    // if (x->idx_type == ALPHA_SPARSE_DCU_INDEXTYPE_I64 && x->data_type == ALPHA_SPARSE_DATATYPE_FLOAT)
    // {
    //     return dcu_s_gthr_i64(handle, x->nnz, (float *)y->values, (float *)x->val_data, (ALPHA_INT *)x->idx_data);
    // }
    // // double real ; i64
    // if (x->idx_type == ALPHA_SPARSE_DCU_INDEXTYPE_I64 && x->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE)
    // {
    //     return dcu_d_gthr_i64(handle, x->nnz, (double *)y->values, (double *)x->val_data, (ALPHA_INT *)x->idx_data);
    // }
    // // single complex ; i64
    // if (x->idx_type == ALPHA_SPARSE_DCU_INDEXTYPE_I64 && x->data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
    // {
    //     return dcu_c_gthr_i64(handle, x->nnz, (ALPHA_Complex8 *)y->values, (ALPHA_Complex8 *)x->val_data, (ALPHA_INT *)x->idx_data);
    // }
    // // double complex ; i64
    // if (x->idx_type == ALPHA_SPARSE_DCU_INDEXTYPE_I64 && x->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
    // {
    //     return dcu_z_gthr_i64(handle, x->nnz, (ALPHA_Complex16 *)y->values, (ALPHA_Complex16 *)x->val_data, (ALPHA_INT *)x->idx_data);
    // }

    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
