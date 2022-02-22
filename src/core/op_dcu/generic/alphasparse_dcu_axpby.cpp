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
alphasparse_dcu_axpby(alphasparse_dcu_handle_t handle,
                      const void *alpha,
                      const alphasparse_dcu_spvec_descr_t x,
                      const void *beta,
                      alphasparse_dcu_dnvec_descr_t y)
{
    // Check for valid handle and matrix descriptor
    if (handle == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
    }

    //
    // Check the rest of pointer arguments
    //
    if (x == nullptr || y == nullptr || alpha == nullptr || beta == nullptr) {
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
        return dcu_s_axpby(handle, alpha, x, beta, y);
    }
    // double real ; i32
    if (x->idx_type == ALPHA_SPARSE_DCU_INDEXTYPE_I32 && x->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE) {
        return dcu_d_axpby(handle, alpha, x, beta, y);
    }
    // single complex ; i32
    if (x->idx_type == ALPHA_SPARSE_DCU_INDEXTYPE_I32 && x->data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
        return dcu_c_axpby(handle, alpha, x, beta, y);
    }
    // double complex ; i32
    if (x->idx_type == ALPHA_SPARSE_DCU_INDEXTYPE_I32 && x->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
        return dcu_z_axpby(handle, alpha, x, beta, y);
    }

    //TODO add support to i64
    // // single real ; i64
    // if (x->idx_type == ALPHA_SPARSE_DCU_INDEXTYPE_I64 && x->data_type == ALPHA_SPARSE_DATATYPE_FLOAT)
    // {
    //     return dcu_s_axpby_i64(handle, alpha, x, beta, y);
    // }
    // // double real ; i64
    // if (x->idx_type == ALPHA_SPARSE_DCU_INDEXTYPE_I64 && x->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE)
    // {
    //     return dcu_d_axpby_i64(handle, alpha, x, beta, y);
    // }
    // // single complex ; i64
    // if (x->idx_type == ALPHA_SPARSE_DCU_INDEXTYPE_I64 && x->data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
    // {
    //     return dcu_c_axpby_i64(handle, alpha, x, beta, y);
    // }
    // // double complex ; i64
    // if (x->idx_type == ALPHA_SPARSE_DCU_INDEXTYPE_I64 && x->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
    // {
    //     return dcu_z_axpby_i64(handle, alpha, x, beta, y);
    // }

    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
