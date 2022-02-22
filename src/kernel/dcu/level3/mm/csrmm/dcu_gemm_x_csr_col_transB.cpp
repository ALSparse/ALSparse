#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/util_dcu.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

__global__ static void
csr_gemm_transB_plain(ALPHA_INT m,
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
    ALPHA_INT tid    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ALPHA_INT stride = hipBlockDim_x * hipGridDim_x;
    for (ALPHA_INT cc = tid; cc < n; cc += stride) {
        for (ALPHA_INT cr = 0; cr < m; ++cr) {
            ALPHA_Number ctmp;
            alpha_setzero(ctmp);
            for (ALPHA_INT ai = csr_row_ptr[cr]; ai < csr_row_ptr[cr + 1]; ++ai) {
                alpha_madde(ctmp, csr_val[ai], x[index2(csr_col_ind[ai], cc, ldx)]);
            }
            alpha_mule(y[index2(cc, cr, ldy)], beta);
            alpha_madde(y[index2(cc, cr, ldy)], alpha, ctmp);
        }
    }
}

template <ALPHA_INT BLOCKSIZE, ALPHA_INT WF_SIZE>
static __global__ void
    __launch_bounds__(BLOCKSIZE)
        csrmmnt_general_device(ALPHA_INT offset,
                               ALPHA_INT ncol,
                               ALPHA_INT M,
                               ALPHA_INT N,
                               ALPHA_INT K,
                               ALPHA_INT nnz,
                               ALPHA_Number alpha,
                               const ALPHA_INT *__restrict__ csr_row_ptr,
                               const ALPHA_INT *__restrict__ csr_col_ind,
                               const ALPHA_Number *__restrict__ csr_val,
                               const ALPHA_Number *__restrict__ matB,
                               ALPHA_INT ldb,
                               ALPHA_Number beta,
                               ALPHA_Number *__restrict__ matC,
                               ALPHA_INT ldc)
{
    ALPHA_INT tid = hipThreadIdx_x;
    ALPHA_INT gid = hipBlockIdx_x * BLOCKSIZE + tid;
    ALPHA_INT row = gid / WF_SIZE;
    ALPHA_INT lid = tid & (WF_SIZE - 1);
    ALPHA_INT wid = tid / WF_SIZE;

    if (row >= M) {
        return;
    }

    ALPHA_Number zero;
    alpha_setzero(zero);

    __shared__ ALPHA_INT shared_col[BLOCKSIZE / WF_SIZE][WF_SIZE];
    __shared__ ALPHA_Number shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE];

    ALPHA_INT row_start = csr_row_ptr[row];
    ALPHA_INT row_end   = csr_row_ptr[row + 1];

    for (ALPHA_INT l = offset; l < ncol; l += WF_SIZE) {
        ALPHA_INT col = l + lid;
        ALPHA_Number sum;
        alpha_setzero(sum);

        for (ALPHA_INT j = row_start; j < row_end; j += WF_SIZE) {
            ALPHA_INT k = j + lid;

            __syncthreads();

            shared_col[wid][lid] = (k < row_end) ? N * (csr_col_ind[k]) : 0;
            shared_val[wid][lid] = (k < row_end) ? csr_val[k] : zero;

            __syncthreads();

            if (col >= ncol) continue;

            for (ALPHA_INT i = 0; i < WF_SIZE; ++i) {
                ALPHA_Number val_B = matB[col + shared_col[wid][i]];
                // sum += shared_val[wid][i] * val_B;
                alpha_madde(sum, shared_val[wid][i], val_B);
            }
        }

        if (col < ncol) {
            // matC[row + col * ldc] = beta * matC[row + col * ldc] + alpha * sum;
            ALPHA_Number tmp;
            alpha_mul(tmp, beta, matC[row + col * ldc]);
            alpha_madd(matC[row + col * ldc], alpha, sum, tmp);
        }
    }
}

template <ALPHA_INT BLOCKSIZE, ALPHA_INT WF_SIZE>
static __global__ void
    __launch_bounds__(BLOCKSIZE)
        csrmmnt_general_unroll2_device(ALPHA_INT offset,
                                       ALPHA_INT ncol,
                                       ALPHA_INT M,
                                       ALPHA_INT N,
                                       ALPHA_INT K,
                                       ALPHA_INT nnz,
                                       ALPHA_Number alpha,
                                       const ALPHA_INT *__restrict__ csr_row_ptr,
                                       const ALPHA_INT *__restrict__ csr_col_ind,
                                       const ALPHA_Number *__restrict__ csr_val,
                                       const ALPHA_Number *__restrict__ matB,
                                       ALPHA_INT ldb,
                                       ALPHA_Number beta,
                                       ALPHA_Number *__restrict__ matC,
                                       ALPHA_INT ldc)
{
    ALPHA_INT tid = hipThreadIdx_x;
    ALPHA_INT gid = hipBlockIdx_x * BLOCKSIZE + tid;
    ALPHA_INT row = gid / WF_SIZE;
    ALPHA_INT lid = tid & (WF_SIZE - 1);
    ALPHA_INT wid = tid / WF_SIZE;

    if (row >= M) {
        return;
    }

    ALPHA_Number zero;
    alpha_setzero(zero);

    __shared__ ALPHA_INT shared_col[BLOCKSIZE / WF_SIZE][WF_SIZE];
    __shared__ ALPHA_Number shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE];

    ALPHA_INT row_start = csr_row_ptr[row];
    ALPHA_INT row_end   = csr_row_ptr[row + 1];

    for (ALPHA_INT l = offset; l < ncol; l += WF_SIZE) {
        ALPHA_INT col = l + lid;
        ALPHA_Number sum;
        alpha_setzero(sum);

        for (ALPHA_INT j = row_start; j < row_end; j += WF_SIZE) {
            ALPHA_INT k = j + lid;

            __syncthreads();

            shared_col[wid][lid] = (k < row_end) ? N * (csr_col_ind[k]) : 0;
            shared_val[wid][lid] = (k < row_end) ? csr_val[k] : zero;

            __syncthreads();

            if (col >= ncol) continue;

            ALPHA_Number sum0, sum1;
            alpha_setzero(sum0);
            alpha_setzero(sum1);
            ALPHA_Number val_B0, val_B1;
            ALPHA_INT i = 0;
            for (; i < WF_SIZE - 1; i += 2) {
                val_B0 = matB[col + shared_col[wid][i]];
                val_B1 = matB[col + shared_col[wid][i + 1]];

                alpha_madde(sum0, shared_val[wid][i], val_B0);
                alpha_madde(sum1, shared_val[wid][i + 1], val_B1);
            }
            for (; i < WF_SIZE; i++) {
                ALPHA_Number val_B = matB[col + shared_col[wid][i]];
                // sum += shared_val[wid][i] * val_B;
                alpha_madde(sum0, shared_val[wid][i], val_B);
            }
            alpha_add(sum, sum, sum0);
            alpha_add(sum, sum, sum1);
        }

        if (col < ncol) {
            // matC[row + col * ldc] = beta * matC[row + col * ldc] + alpha * sum;
            ALPHA_Number tmp;
            alpha_mul(tmp, beta, matC[row + col * ldc]);
            alpha_madd(matC[row + col * ldc], alpha, sum, tmp);
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
    // const ALPHA_INT threadPerBlock = 256;
    // const int blockPerGrid = min(32, (threadPerBlock + m - 1) / threadPerBlock);

    // hipLaunchKernelGGL(csr_gemm_transB_plain, dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream,
    //                    m, n, k, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind,
    //                    matB, ldb, beta, matC, ldc);

    const ALPHA_INT WF_SIZE   = 64;
    const ALPHA_INT BLOCKSIZE = 512;
    hipLaunchKernelGGL((csrmmnt_general_device<BLOCKSIZE, WF_SIZE>), dim3((WF_SIZE * m - 1) / BLOCKSIZE + 1), dim3(BLOCKSIZE), 0, handle->stream, 0, n, m, n, k, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, matB, ldb, beta, matC, ldc);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
