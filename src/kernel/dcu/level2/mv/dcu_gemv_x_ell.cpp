#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"



template <ALPHA_INT BLOCKSIZE>
__global__ static void
    __launch_bounds__(BLOCKSIZE)
        ell_gemv_unrolln(ALPHA_INT m,
                         ALPHA_INT n,
                         const ALPHA_Number alpha,
                         const ALPHA_Number *ell_val,
                         const ALPHA_INT *ell_col_ind,
                         ALPHA_INT ell_width,
                         const ALPHA_Number *x,
                         const ALPHA_Number beta,
                         ALPHA_Number *y)
{
    ALPHA_INT ar = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (ar >= m) return;

    ALPHA_Number sum;
    alpha_setzero(sum);

    for (ALPHA_INT i = 0; i < ell_width; i++) {
        ALPHA_INT idx = m * i + ar;
        ALPHA_INT col = ell_col_ind[idx];
        if (col < n) {
            // sum += ell_val[idx] * x[col];
            alpha_madde(sum, ell_val[idx], x[col]);
        }
    }

    ALPHA_Number left, right;
    alpha_mul(left, beta, y[ar]);
    alpha_mul(right, alpha, sum);
    alpha_add(y[ar], left, right);
}

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      ALPHA_INT m,
      ALPHA_INT n,
      const ALPHA_Number alpha,
      const ALPHA_Number *ell_val,
      const ALPHA_INT *ell_col_ind,
      ALPHA_INT ell_width,
      const ALPHA_Number *x,
      const ALPHA_Number beta,
      ALPHA_Number *y)
{
    const ALPHA_INT nnz_per_row = ell_width;

    const ALPHA_INT BLOCKSIZE      = 512;
    const ALPHA_INT threadPerBlock = BLOCKSIZE;
    const ALPHA_INT blockPerGrid   = (m - 1) / BLOCKSIZE + 1;

    hipLaunchKernelGGL((ell_gemv_unrolln<BLOCKSIZE>), blockPerGrid, threadPerBlock, 0, handle->stream, m, n, alpha, ell_val, ell_col_ind, ell_width, x, beta, y);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
