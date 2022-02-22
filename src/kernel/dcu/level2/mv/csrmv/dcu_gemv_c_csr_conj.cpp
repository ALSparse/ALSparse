#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/common_dcu.h"

#include "dcu_gemv_common.h"

__global__ static void
beta_y(ALPHA_INT m,
       ALPHA_Number beta,
       ALPHA_Number *y)
{
    ALPHA_INT row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (row >= m) return;

    alpha_mul(y[row], y[row], beta);
}

__global__ static void
csr_gemv_conj_scalar_atomicadd(ALPHA_INT m,
                               ALPHA_INT n,
                               ALPHA_INT nnz,
                               const ALPHA_Number alpha,
                               const ALPHA_Number *csr_val,
                               const ALPHA_INT *csr_row_ptr,
                               const ALPHA_INT *csr_col_ind,
                               const ALPHA_Number *x,
                               const ALPHA_Number beta,
                               ALPHA_Number *y)
{
    ALPHA_INT row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (row >= m) return;

    ALPHA_Number xval = x[row];

    for (ALPHA_INT j = csr_row_ptr[row]; j < csr_row_ptr[row + 1]; j++) {
        ALPHA_Number val = csr_val[j];
        ALPHA_INT col    = csr_col_ind[j];

        alpha_conj(val, val);
        alpha_mul(val, val, alpha);
        alpha_mul(val, val, xval);
        alpha_atomic_add(y[col], val);
    }
}

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      ALPHA_INT m,
      ALPHA_INT n,
      ALPHA_INT nnz,
      const ALPHA_Number alpha,
      const ALPHA_Number *csr_val,
      const ALPHA_INT *csr_row_ptr,
      const ALPHA_INT *csr_col_ind,
      alphasparse_dcu_mat_info_t info,
      const ALPHA_Number *x,
      const ALPHA_Number beta,
      ALPHA_Number *y)
{
    const ALPHA_INT threadPerBlock = 512;
    const ALPHA_INT blockPerGrid   = m / threadPerBlock + 1;
    hipLaunchKernelGGL((beta_y), dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, m, beta, y);
    hipLaunchKernelGGL((csr_gemv_conj_scalar_atomicadd), dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
