#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/util_dcu.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

__global__ static void
csr_gemmi_plain(ALPHA_INT m,
                ALPHA_INT n,
                ALPHA_INT k,
                ALPHA_INT nnz,
                const ALPHA_Number alpha,
                const ALPHA_Number *matA,
                ALPHA_INT lda,
                const ALPHA_Number *csr_val,
                const ALPHA_INT *csr_row_ptr,
                const ALPHA_INT *csr_col_ind,
                const ALPHA_Number beta,
                ALPHA_Number *matC,
                ALPHA_INT ldc)
{
    ALPHA_INT tid    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ALPHA_INT stride = hipBlockDim_x * hipGridDim_x;

    for (ALPHA_INT i = tid; i < m; i += stride) {
        for (ALPHA_INT j = 0; j < n; j++) {
            alpha_mul(matC[index2(j, i, ldc)], matC[index2(j, i, ldc)], beta);
        }
    }

    for (ALPHA_INT i = tid; i < m; i += stride) // 如果按k划分任务，存在写冲突
    {
        for (ALPHA_INT j = 0; j < k; j++) {
            for (ALPHA_INT bi = csr_row_ptr[j]; bi < csr_row_ptr[j + 1]; bi++) {
                ALPHA_INT bc    = csr_col_ind[bi];
                ALPHA_Number bv = csr_val[bi];
                alpha_mul(bv, bv, alpha);
                alpha_madde(matC[index2(bc, i, ldc)], matA[index2(j, i, lda)], bv);
            }
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      ALPHA_INT m,
      ALPHA_INT n,
      ALPHA_INT k,
      ALPHA_INT nnz,
      const ALPHA_Number alpha,
      const ALPHA_Number *A,
      ALPHA_INT lda,
      const ALPHA_Number *csr_val,
      const ALPHA_INT *csr_row_ptr,
      const ALPHA_INT *csr_col_ind,
      const ALPHA_Number beta,
      ALPHA_Number *matC,
      ALPHA_INT ldc)
{
    const ALPHA_INT threadPerBlock = 256;
    const int blockPerGrid         = (threadPerBlock + m - 1) / threadPerBlock;

    hipLaunchKernelGGL(csr_gemmi_plain, dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, m, n, k, nnz, alpha, A, lda, csr_val, csr_row_ptr, csr_col_ind, beta, matC, ldc);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
