#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

__global__ static void spts_syncfree_cuda_executor_csr_wrt_thread(const ALPHA_INT *csrRowPtr,
                                                                  const ALPHA_INT *csrColIdx,
                                                                  const ALPHA_Number *csrVal,
                                                                  volatile ALPHA_INT *get_value,
                                                                  const ALPHA_INT m,
                                                                  const ALPHA_INT nnz,
                                                                  const ALPHA_Number alpha,
                                                                  const ALPHA_Number *x,
                                                                  ALPHA_Number *y,
                                                                  ALPHA_Number *diag)

{
    const ALPHA_INT global_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const ALPHA_INT stride    = hipBlockDim_x * hipGridDim_x;

    // TODO:opt here
    // diag = csrVal[csrRotPtr[row+1]-1];
    for (ALPHA_INT r = global_id; r < m; r += stride) {
        for (ALPHA_INT ai = csrRowPtr[r]; ai < csrRowPtr[r + 1]; ai++) {
            ALPHA_INT ac = csrColIdx[ai];
            if (ac == r) {
                diag[r] = csrVal[ai];
            }
        }
    }

    ALPHA_INT col, j, repeat = 0;
    ALPHA_Number yi;
    ALPHA_Number left_sum;
    alpha_setzero(left_sum);

    for (ALPHA_INT i = global_id; i < m; i += stride) {
        alpha_setzero(left_sum);
        j = csrRowPtr[i];
        while (j < csrRowPtr[i + 1]) {
            col = csrColIdx[j];

            while (get_value[col] == 1) {
                if (col < i) {
                    alpha_madde(left_sum, csrVal[j], y[col]);
                } else
                    break;
                j++;
                col = csrColIdx[j];
            }

            ALPHA_INT tmp_try = !(i ^ col);
            {
                ALPHA_Number tmp;
                alpha_mul(tmp, alpha, x[i]);
                alpha_sub(tmp, tmp, left_sum);
                alpha_div(yi, tmp, diag[i]);
                alpha_cross_entropy(y[i], yi, y[i], tmp_try);
                get_value[i] = tmp_try | get_value[i];
                __threadfence();

                if (tmp_try)
                    break;
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
      ALPHA_INT nnz,
      const ALPHA_Number alpha,
      const ALPHA_Number *csr_val,
      const ALPHA_INT *csr_row_ptr,
      const ALPHA_INT *csr_col_ind,
      alphasparse_dcu_mat_info_t info,
      const ALPHA_Number *x,
      ALPHA_Number *y,
      alphasparse_dcu_solve_policy_t policy,
      void *temp_buffer)
{
    ALPHA_Number *diag;
    hipMalloc((void **)&diag, sizeof(ALPHA_Number) * m);
    hipMemset(diag, '\0', sizeof(ALPHA_Number) * m);

    const ALPHA_INT threadPerBlock = 256;
    const int blockPerGrid         = (m - 1) / threadPerBlock + 1;

    ALPHA_INT *get_value;
    hipMalloc((void **)&get_value, (m) * sizeof(ALPHA_INT));
    hipMemset(get_value, 0, sizeof(ALPHA_INT) * m);
    hipLaunchKernelGGL(spts_syncfree_cuda_executor_csr_wrt_thread, dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, csr_row_ptr, csr_col_ind, csr_val, get_value, m, nnz, alpha, x, y, diag);

    hipFree(get_value);
    hipFree(diag);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
