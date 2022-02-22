#include "dcu_gemv_common.h"
#include "dcu_gemv_vector.h"

#ifdef __cplusplus
extern "C" {
#endif
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
    alphasparse_status_t st = ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    u_int32_t flag = 0;

    csrgemv_algo algo = VECTOR;

    const ALPHA_INT nnz_per_row = nnz / m;

    if (algo == VECTOR) {
        st = csr_gemv_vector_dispatch(handle, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y, flag);
    }
    return st;
}
#undef BLOCKSIZE
#undef WF_SIZE

#ifdef __cplusplus
}
#endif
