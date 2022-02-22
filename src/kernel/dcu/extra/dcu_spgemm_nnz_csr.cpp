#include "alphasparse/handle.h"
#include <hip/hip_runtime.h>
#include "alphasparse/util_dcu.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

__global__ static void
gemm_nnz_per_row(ALPHA_INT m,
                 ALPHA_INT n,
                 ALPHA_INT k,
                 const ALPHA_INT *csr_row_ptr_A,
                 const ALPHA_INT *csr_col_ind_A,
                 const ALPHA_INT *csr_row_ptr_B,
                 const ALPHA_INT *csr_col_ind_B,
                 const ALPHA_INT *csr_row_ptr_D,
                 const ALPHA_INT *csr_col_ind_D,
                 ALPHA_INT *row_nnz)
{
    ALPHA_INT tid    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ALPHA_INT stride = hipBlockDim_x * hipGridDim_x;

    const ALPHA_INT trunk_size = 2048;
    bool flag[trunk_size];
    ALPHA_INT trunk = 0;
    for (ALPHA_INT ar = tid; ar < m; ar += stride) {
        while (trunk < n) {
            for (ALPHA_INT i = 0; i < trunk_size; i++)
                flag[i] = false;

            for (ALPHA_INT di = csr_row_ptr_D[ar]; di < csr_row_ptr_D[ar + 1]; di++) {
                ALPHA_INT dc = csr_col_ind_D[di];
                if (dc >= trunk && dc < trunk + trunk_size) {
                    flag[dc - trunk] = true;
                    row_nnz[ar + 1]++;
                }
            }

            for (ALPHA_INT ai = csr_row_ptr_A[ar]; ai < csr_row_ptr_A[ar + 1]; ai++) {
                ALPHA_INT br = csr_col_ind_A[ai];
                for (ALPHA_INT bi = csr_row_ptr_B[br]; bi < csr_row_ptr_B[br + 1]; bi++) {
                    ALPHA_INT bc = csr_col_ind_B[bi];
                    if (bc >= trunk && bc < trunk + trunk_size && !flag[bc - trunk]) {
                        row_nnz[ar + 1]++;
                        flag[bc - trunk] = true;
                    }
                }
            }
            trunk += trunk_size;
        }
    }
}

__global__ static void
prefix(ALPHA_INT *row_nnz, ALPHA_INT size)
{
    // todo: opt, add a opt prefix to util
    ALPHA_INT tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (tid == 0)
        for (int i = 1; i < size; i++) {
            row_nnz[i] += row_nnz[i - 1];
            //printf("%d, %d\n", i, row_nnz[i]);
        }
}

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

alphasparse_status_t
dcu_spgemm_nnz_csr(alphasparse_dcu_handle_t handle,
                   ALPHA_INT m,
                   ALPHA_INT n,
                   ALPHA_INT k,
                   ALPHA_INT nnz_A,
                   const ALPHA_INT *csr_row_ptr_A,
                   const ALPHA_INT *csr_col_ind_A,
                   ALPHA_INT nnz_B,
                   const ALPHA_INT *csr_row_ptr_B,
                   const ALPHA_INT *csr_col_ind_B,
                   ALPHA_INT nnz_D,
                   const ALPHA_INT *csr_row_ptr_D,
                   const ALPHA_INT *csr_col_ind_D,
                   ALPHA_INT *csr_row_ptr_C,
                   ALPHA_INT *nnz_C,
                   const alphasparse_dcu_mat_info_t info_C,
                   void *temp_buffer)
{
    const ALPHA_INT threadPerBlock = 256;
    const int blockPerGrid         = (m - 1) / threadPerBlock + 1;

    hipMemset(csr_row_ptr_C, '\0', (m + 1) * sizeof(ALPHA_INT));

    hipLaunchKernelGGL(gemm_nnz_per_row, blockPerGrid, threadPerBlock, 0, handle->stream, m, n, k, csr_row_ptr_A, csr_col_ind_A, csr_row_ptr_B, csr_col_ind_B, csr_row_ptr_D, csr_col_ind_D, csr_row_ptr_C);

    hipLaunchKernelGGL(prefix, dim3(1), dim3(1), 0, handle->stream, csr_row_ptr_C, m + 1);

    hipMemcpyAsync(nnz_C, csr_row_ptr_C + m, sizeof(ALPHA_INT), hipMemcpyDeviceToHost);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
