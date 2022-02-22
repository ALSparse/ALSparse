#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

const ALPHA_INT threadPerBlock = 256;

__global__ static void
roti(ALPHA_INT nnz,
     ALPHA_Number *x_val,
     const ALPHA_INT *x_ind,
     ALPHA_Number *y,
     ALPHA_Number c,
     ALPHA_Number s)
{
    int tid    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (ALPHA_INT i = tid; i < nnz; i += stride) {
        ALPHA_Number x_tmp = x_val[i];
        ALPHA_Number y_tmp = y[x_ind[i]];

        // x_val[i] = c * x_tmp + s * y_tmp;
        // y[x_ind[i]] = c * y_tmp - s * x_tmp;
        ALPHA_Number left, right;
        alpha_mul(left, c, x_tmp);
        alpha_mul(right, s, y_tmp);
        alpha_add(x_val[i], left, right);

        alpha_mul(left, c, y_tmp);
        alpha_mul(right, s, x_tmp);
        alpha_sub(y[x_ind[i]], left, right);
    }
}

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      ALPHA_INT nnz,
      ALPHA_Number *x_val,
      const ALPHA_INT *x_ind,
      ALPHA_Number *y,
      const ALPHA_Number *c,
      const ALPHA_Number *s)
{
    const int blockPerGrid = min(2, (threadPerBlock + nnz - 1) / threadPerBlock);

    hipLaunchKernelGGL(roti, dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, nnz, x_val, x_ind, y, *c, *s);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
