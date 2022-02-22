#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/util_dcu.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "alphasparse/kernel_dcu.h"

alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      ALPHA_INT m,
      ALPHA_INT nrhs,
      ALPHA_INT nnz,
      const ALPHA_Number alpha,
      const ALPHA_Number *csr_val,
      const ALPHA_INT *csr_row_ptr,
      const ALPHA_INT *csr_col_ind,
      ALPHA_Number *B,
      ALPHA_INT ldb,
      alphasparse_dcu_mat_info_t info,
      alphasparse_dcu_solve_policy_t policy,
      void *temp_buffer)
{
    dcu_trsm_csr_u_lo(handle, m, nrhs, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, B, ldb, info, policy, temp_buffer);
    // hipLaunchKernelGGL(csr_gemm_plain, dim3(2), dim3(128), 0, handle->stream,
    //                    m, n, k, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind,
    //                    matB, ldb, beta, matC, ldc);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
