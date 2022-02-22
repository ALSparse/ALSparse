#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

const ALPHA_INT threadPerBlock = 256;

__global__ static void
gthr(ALPHA_INT nnz,
     const ALPHA_Number *x_val,
     const ALPHA_INT *x_ind,
     ALPHA_Number *y)
{
    int tid    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (ALPHA_INT i = tid; i < nnz; i += stride) {
        y[x_ind[i]] = x_val[i];
    }
}

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      ALPHA_INT nnz,
      const ALPHA_Number *x_val,
      const ALPHA_INT *x_ind,
      ALPHA_Number *y)
{
    const int blockPerGrid = min(2, (threadPerBlock + nnz - 1) / threadPerBlock);

    hipLaunchKernelGGL(gthr, dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, nnz, x_val, x_ind, y);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
