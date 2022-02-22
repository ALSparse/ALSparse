#include "dcu_gemm_csr_col_plain.h"
// #include "dcu_gemm_csr_col_block.h"

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

    const ALPHA_INT BLOCKSIZE = 256;
    const ALPHA_INT WF_SIZE   = 8;

    gemm_col_plain_dispatch(handle, m, n, k, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, matB, ldb, beta, matC, ldc);
    // gemm_col_block_dispatch(handle, m, n, k, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, matB, ldb, beta, matC, ldc);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
