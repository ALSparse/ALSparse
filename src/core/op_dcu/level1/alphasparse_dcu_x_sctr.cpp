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
ONAME(alphasparse_dcu_handle_t handle,
      ALPHA_INT nnz,
      const ALPHA_Number *x_val,
      const ALPHA_INT *x_ind,
      ALPHA_Number *y,
      alphasparse_index_base_t idx_base)
{
    // Check for valid handle and matrix descriptor
    if (handle == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
    }

    // Check index base
    if (idx_base != ALPHA_SPARSE_INDEX_BASE_ZERO && idx_base != ALPHA_SPARSE_INDEX_BASE_ONE) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Check sizes
    if (nnz < 0) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return if possible
    if (nnz == 0) {
        return ALPHA_SPARSE_STATUS_SUCCESS;
    }

    //
    // Check the rest of pointer arguments
    //
    if (x_val == nullptr || x_ind == nullptr || y == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    return dcu_sctr(handle, nnz, x_val, x_ind, y);
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
