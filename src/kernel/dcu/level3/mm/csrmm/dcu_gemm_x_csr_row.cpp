#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/util_dcu.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

// #include "dcu_gemm_csr_row_block.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      ALPHA_INT m,
      ALPHA_INT n,
      ALPHA_INT k,
      ALPHA_INT nnz,
      ALPHA_Number alpha,
      const ALPHA_Number *csr_val,
      const ALPHA_INT *csr_row_ptr,
      const ALPHA_INT *csr_col_ind,
      const ALPHA_Number *matB,
      ALPHA_INT ldb,
      ALPHA_Number beta,
      ALPHA_Number *matC,
      ALPHA_INT ldc)
{
    alphasparse_status_t st;

    // gemm_csr_row_block_dispatch(handle, m, n, k, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, matB, ldb, beta, matC, ldc);

    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
