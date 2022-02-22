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
                                                                  ALPHA_Number *y)
{
    const ALPHA_INT global_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const ALPHA_INT stride    = hipBlockDim_x * hipGridDim_x;

    ALPHA_INT row_end, col_end;
    ALPHA_INT col, j;
    ALPHA_Number yi;
    ALPHA_Number left_sum;
    alpha_setzero(left_sum);

    for (ALPHA_INT i = global_id; i < m; i += stride) {
        row_end = csrRowPtr[i + 1];

        alpha_setzero(left_sum);
        j = csrRowPtr[i];

        col_end = csrColIdx[row_end - 1];

        y[i] = alpha; // init yi for nnz_row==0 case

        while (j < csrRowPtr[i + 1]) {
            col       = csrColIdx[j];
            bool flag = false; // 记录是否到达对角线，当csr存储的对角线没有元素时

            while (get_value[col] == 1) {
                if (col < i && col <= col_end) {
                    alpha_madde(left_sum, csrVal[j], y[col]);
                }

                if (col >= col_end) {
                    flag = true;
                    break;
                }
                j++;
                col = csrColIdx[j];
            }

            ALPHA_INT tmp_try = (!(i ^ col)) | (flag) | (col > i);
            {
                ALPHA_Number tmp;
                alpha_mul(tmp, alpha, x[i]);
                alpha_sub(yi, tmp, left_sum);
                alpha_cross_entropy(y[i], yi, y[i], tmp_try);
                get_value[i] = tmp_try | get_value[i];
                __threadfence();

                if (tmp_try)
                    break;
            }
        }
        get_value[i] = 1; // tag get_value for nnz_row==0 case
        __threadfence();
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
    const ALPHA_INT threadPerBlock = 256;
    const ALPHA_INT blockPerGrid   = (m - 1) / threadPerBlock + 1;

    ALPHA_INT *get_value;
    hipMalloc((void **)&get_value, (m) * sizeof(ALPHA_INT));
    hipMemset(get_value, 0, sizeof(ALPHA_INT) * m);
    hipLaunchKernelGGL(spts_syncfree_cuda_executor_csr_wrt_thread, dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, csr_row_ptr, csr_col_ind, csr_val, get_value, m, nnz, alpha, x, y);
    hipFree(get_value);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
