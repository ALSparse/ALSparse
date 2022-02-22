#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/util_dcu.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

__global__ static void
csr_gemm_plain_device(ALPHA_INT m,
                      ALPHA_INT n,
                      ALPHA_INT k,
                      ALPHA_INT nnz,
                      ALPHA_Number alpha,
                      const ALPHA_Number *csr_val,
                      const ALPHA_INT *csr_row_ptr,
                      const ALPHA_INT *csr_col_ind,
                      const ALPHA_Number *x,
                      ALPHA_INT ldx,
                      ALPHA_Number beta,
                      ALPHA_Number *y,
                      ALPHA_INT ldy)
{
    ALPHA_INT bid      = hipBlockIdx_x;
    ALPHA_INT b_stride = hipGridDim_x;
    ALPHA_INT tid      = hipThreadIdx_x;
    ALPHA_INT t_stride = hipBlockDim_x;
    for (ALPHA_INT cc = bid; cc < n; cc += b_stride) {
        for (ALPHA_INT cr = tid; cr < m; cr += t_stride) {
            ALPHA_Number ctmp;
            alpha_setzero(ctmp);
            for (ALPHA_INT ai = csr_row_ptr[cr]; ai < csr_row_ptr[cr + 1]; ++ai) {
                alpha_madde(ctmp, csr_val[ai], x[index2(cc, csr_col_ind[ai], ldx)]);
            }
            alpha_mule(y[index2(cc, cr, ldy)], beta);
            alpha_madde(y[index2(cc, cr, ldy)], alpha, ctmp);
        }
    }
}

static alphasparse_status_t gemm_col_plain_dispatch(alphasparse_dcu_handle_t handle,
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
    const int threadPerBlock = 256;
    const int blockPerGrid   = min(32, (threadPerBlock + k - 1) / threadPerBlock);
    hipLaunchKernelGGL(csr_gemm_plain_device, dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, m, n, k, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, matB, ldb, beta, matC, ldc);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}