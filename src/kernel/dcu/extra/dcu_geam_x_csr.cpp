#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/util_dcu.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

__global__ static void
add_plain(ALPHA_INT m,
          const ALPHA_Number alpha,
          const ALPHA_Number *csr_val_A,
          const ALPHA_INT *csr_row_ptr_A,
          const ALPHA_INT *csr_col_ind_A,
          const ALPHA_Number beta,
          const ALPHA_Number *csr_val_B,
          const ALPHA_INT *csr_row_ptr_B,
          const ALPHA_INT *csr_col_ind_B,
          ALPHA_Number *csr_val_C,
          const ALPHA_INT *csr_row_ptr_C,
          ALPHA_INT *csr_col_ind_C)
{
    ALPHA_INT tid    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ALPHA_INT stride = hipBlockDim_x * hipGridDim_x;

    for (ALPHA_INT r = tid; r < m; r += stride) {
        ALPHA_INT ai = csr_row_ptr_A[r];
        ALPHA_INT ae = csr_row_ptr_A[r + 1];
        ALPHA_INT bi = csr_row_ptr_B[r];
        ALPHA_INT be = csr_row_ptr_B[r + 1];

        ALPHA_INT ci = csr_row_ptr_C[r];
        ALPHA_Number tmp1, tmp2;
        while (ai < ae && bi < be) {
            ALPHA_INT ac = csr_col_ind_A[ai];
            ALPHA_INT bc = csr_col_ind_B[bi];
            if (ac < bc) {
                csr_col_ind_C[ci] = ac;
                alpha_mul(csr_val_C[ci], alpha, csr_val_A[ai]);
                ai++;
            } else if (ac > bc) {
                csr_col_ind_C[ci] = bc;
                alpha_mul(csr_val_C[ci], beta, csr_val_B[bi]);
                bi++;
            } else {
                csr_col_ind_C[ci] = bc;
                alpha_mul(tmp1, alpha, csr_val_A[ai]);
                alpha_mul(tmp2, beta, csr_val_B[bi]);
                // csr_val_C[ci] = alpha * csr_val_A[ai] + beta * csr_val_B[bi];
                alpha_add(csr_val_C[ci], tmp1, tmp2);
                ai++;
                bi++;
            }
            ci++;
        }
        if (ai == ae) {
            for (; bi < be; bi++, ci++) {
                csr_col_ind_C[ci] = csr_col_ind_B[bi];
                alpha_mul(csr_val_C[ci], beta, csr_val_B[bi]);
            }
        } else {
            for (; ai < ae; ai++, ci++) {
                csr_col_ind_C[ci] = csr_col_ind_A[ai];
                alpha_mul(csr_val_C[ci], alpha, csr_val_A[ai]);
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
      const ALPHA_Number alpha,
      ALPHA_INT nnz_A,
      const ALPHA_Number *csr_val_A,
      const ALPHA_INT *csr_row_ptr_A,
      const ALPHA_INT *csr_col_ind_A,
      const ALPHA_Number beta,
      ALPHA_INT nnz_B,
      const ALPHA_Number *csr_val_B,
      const ALPHA_INT *csr_row_ptr_B,
      const ALPHA_INT *csr_col_ind_B,
      ALPHA_Number *csr_val_C,
      const ALPHA_INT *csr_row_ptr_C,
      ALPHA_INT *csr_col_ind_C)
{
    const ALPHA_INT threadPerBlock = 256;
    const int blockPerGrid         = min(32, (threadPerBlock + n - 1) / threadPerBlock);

    hipLaunchKernelGGL(add_plain, dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, m, alpha, csr_val_A, csr_row_ptr_A, csr_col_ind_A, beta, csr_val_B, csr_row_ptr_B, csr_col_ind_B, csr_val_C, csr_row_ptr_C, csr_col_ind_C);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
