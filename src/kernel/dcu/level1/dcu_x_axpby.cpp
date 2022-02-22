#include <hip/hip_runtime.h>
#include "alphasparse/handle.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

const ALPHA_INT threadPerBlock = 256;

__global__ static void
ybeta(ALPHA_Number beta,
      ALPHA_INT size,
      ALPHA_Number *y)
{
    int tid    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (ALPHA_INT i = tid; i < size; i += stride) {
        alpha_mul(y[i], beta, y[i]);
    }
}

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "alphasparse/kernel_dcu.h"

alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      const void *alpha,
      const alphasparse_dcu_spvec_descr_t x,
      const void *beta,
      alphasparse_dcu_dnvec_descr_t y)
{
    const ALPHA_INT nnz          = x->nnz;
    const ALPHA_INT size         = y->size;
    const ALPHA_INT blockPerGrid = min(2, (threadPerBlock + nnz - 1) / threadPerBlock);
    ALPHA_Number lbeta           = *(ALPHA_Number *)beta;
    ALPHA_Number lalpha          = *(ALPHA_Number *)alpha;

    hipLaunchKernelGGL(ybeta, dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, lbeta, size, (ALPHA_Number *)y->values);

    hipDeviceSynchronize();

    dcu_axpyi(handle, nnz, lalpha, (ALPHA_Number *)x->val_data, (ALPHA_INT *)x->idx_data, (ALPHA_Number *)y->values);

    // hipLaunchKernelGGL(axpyi, dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream,
    //                    nnz, lalpha, (ALPHA_Number *)x->val_data, (ALPHA_INT *)x->idx_data, (ALPHA_Number *)y->values);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
