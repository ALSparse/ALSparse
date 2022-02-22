#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/util_dcu.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

__global__ static void
csr_gemmi_trans_plain(ALPHA_INT m,
                      ALPHA_INT n,
                      ALPHA_INT k,
                      ALPHA_INT nnz,
                      const ALPHA_Number alpha,
                      const ALPHA_Number *matA,
                      ALPHA_INT lda,
                      const ALPHA_Number *csr_val,
                      const ALPHA_INT *csr_row_ptr,
                      const ALPHA_INT *csr_col_ind,
                      const ALPHA_Number beta,
                      ALPHA_Number *matC,
                      ALPHA_INT ldc,
                      ALPHA_Number *tmpC)
{
    ALPHA_INT tid    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ALPHA_INT stride = hipBlockDim_x * hipGridDim_x;

    for (ALPHA_INT i = tid; i < n; i += stride) {
        for (ALPHA_INT j = 0; j < m; j++) {
            alpha_mul(matC[index2(i, j, ldc)], matC[index2(i, j, ldc)], beta);
        }
    }

    for (ALPHA_INT i = tid; i < n; i += stride) //for matB cols
    {
        for (ALPHA_INT bi = csr_row_ptr[i]; bi < csr_row_ptr[i + 1]; bi++) {
            ALPHA_INT bc    = csr_col_ind[bi];
            ALPHA_Number bv = csr_val[bi];
            alpha_mul(bv, bv, alpha);
            for (ALPHA_INT j = 0; j < m; j++) {
                alpha_madde(tmpC[index2(i, j, ldc)], matA[index2(bc, j, lda)], bv);
            }
        }
    }

    for (ALPHA_INT i = tid; i < n; i += stride) {
        for (ALPHA_INT j = 0; j < m; j++) {
            alpha_add(matC[index2(i, j, ldc)], matC[index2(i, j, ldc)], tmpC[index2(i, j, ldc)]);
        }
    }
}

template <ALPHA_INT BLOCKSIZE>
__global__ static void gemmit_kernel(ALPHA_INT m,
                                     ALPHA_Number alpha,
                                     const ALPHA_Number *__restrict__ A,
                                     ALPHA_INT lda,
                                     const ALPHA_INT *__restrict__ csr_row_ptr,
                                     const ALPHA_INT *__restrict__ csr_col_ind,
                                     const ALPHA_Number *__restrict__ csr_val,
                                     ALPHA_Number beta,
                                     ALPHA_Number *__restrict__ matC,
                                     ALPHA_INT ldc)
{
    ALPHA_INT row = hipBlockIdx_y;
    ALPHA_INT col = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

    // Do not run out of bounds
    if (col >= m) {
        return;
    }

    // Row entry into B
    ALPHA_INT row_begin = csr_row_ptr[row];
    ALPHA_INT row_end   = csr_row_ptr[row + 1];

    // Accumulator
    ALPHA_Number sum;
    ALPHA_INT col_B;
    ALPHA_Number val_B;
    ALPHA_Number val_A;

    alpha_setzero(sum);

    // Loop over the column indices of B of the current row
    for (ALPHA_INT k = row_begin; k < row_end; ++k) {
        ALPHA_INT col_B    = csr_col_ind[k];
        ALPHA_Number val_B = csr_val[k];
        ALPHA_Number val_A = A[col_B * lda + col];

        // sum += val_A * val_B;
        alpha_madde(sum, val_A, val_B);
    }

    // Write result back to matC
    // matC[row * ldc + col] = alpha * sum + beta * matC[row * ldc + col];
    ALPHA_Number tmp1, tmp2;
    alpha_mul(tmp1, alpha, sum);
    alpha_mul(tmp2, beta, matC[row * ldc + col]);
    alpha_add(matC[row * ldc + col], tmp1, tmp2);
}

template <ALPHA_INT BLOCKSIZE>
__global__ static void gemmit_unroll4_kernel(ALPHA_INT m,
                                             ALPHA_Number alpha,
                                             const ALPHA_Number *__restrict__ A,
                                             ALPHA_INT lda,
                                             const ALPHA_INT *__restrict__ csr_row_ptr,
                                             const ALPHA_INT *__restrict__ csr_col_ind,
                                             const ALPHA_Number *__restrict__ csr_val,
                                             ALPHA_Number beta,
                                             ALPHA_Number *__restrict__ matC,
                                             ALPHA_INT ldc)
{
    ALPHA_INT row = hipBlockIdx_y;
    ALPHA_INT col = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

    // Do not run out of bounds
    if (col >= m) {
        return;
    }

    // Row entry into B
    ALPHA_INT row_begin = csr_row_ptr[row];
    ALPHA_INT row_end   = csr_row_ptr[row + 1];

    // Accumulator
    ALPHA_Number sum0, sum1, sum2, sum3;
    ALPHA_INT col_B0, col_B1, col_B2, col_B3;
    ALPHA_Number val_B0, val_B1, val_B2, val_B3;
    ALPHA_Number val_A0, val_A1, val_A2, val_A3;

    alpha_setzero(sum0);
    alpha_setzero(sum1);
    alpha_setzero(sum2);
    alpha_setzero(sum3);

    // Loop over the column indices of B of the current row
    ALPHA_INT k = row_begin;
    for (; k < row_end - 3; k += 4) {
        col_B0 = csr_col_ind[k];
        col_B1 = csr_col_ind[k + 1];
        col_B2 = csr_col_ind[k + 2];
        col_B3 = csr_col_ind[k + 3];

        val_B0 = csr_val[k];
        val_B1 = csr_val[k + 1];
        val_B2 = csr_val[k + 2];
        val_B3 = csr_val[k + 3];

        val_A0 = A[col_B0 * lda + col];
        val_A1 = A[col_B1 * lda + col];
        val_A2 = A[col_B2 * lda + col];
        val_A3 = A[col_B3 * lda + col];

        alpha_madde(sum0, val_A0, val_B0);
        alpha_madde(sum1, val_A1, val_B1);
        alpha_madde(sum2, val_A2, val_B2);
        alpha_madde(sum3, val_A3, val_B3);
    }
    for (; k < row_end; ++k) {
        ALPHA_INT col_B    = csr_col_ind[k];
        ALPHA_Number val_B = csr_val[k];
        ALPHA_Number val_A = A[col_B * lda + col];

        // sum += val_A * val_B;
        alpha_madde(sum0, val_A, val_B);
    }
    alpha_add(sum0, sum0, sum1);
    alpha_add(sum2, sum2, sum3);
    alpha_add(sum0, sum0, sum2);

    // Write result back to matC
    // matC[row * ldc + col] = alpha * sum + beta * matC[row * ldc + col];
    ALPHA_Number tmp1, tmp2;
    alpha_mul(tmp1, alpha, sum0);
    alpha_mul(tmp2, beta, matC[row * ldc + col]);
    alpha_add(matC[row * ldc + col], tmp1, tmp2);
}

template <ALPHA_INT BLOCKSIZE>
__global__ static void gemmit_unroll2_kernel(ALPHA_INT m,
                                             ALPHA_Number alpha,
                                             const ALPHA_Number *__restrict__ A,
                                             ALPHA_INT lda,
                                             const ALPHA_INT *__restrict__ csr_row_ptr,
                                             const ALPHA_INT *__restrict__ csr_col_ind,
                                             const ALPHA_Number *__restrict__ csr_val,
                                             ALPHA_Number beta,
                                             ALPHA_Number *__restrict__ matC,
                                             ALPHA_INT ldc)
{
    ALPHA_INT row = hipBlockIdx_y;
    ALPHA_INT col = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

    // Do not run out of bounds
    if (col >= m) {
        return;
    }

    // Row entry into B
    ALPHA_INT row_begin = csr_row_ptr[row];
    ALPHA_INT row_end   = csr_row_ptr[row + 1];

    // Accumulator
    ALPHA_Number sum0, sum1, sum2, sum3;
    ALPHA_INT col_B0, col_B1, col_B2, col_B3;
    ALPHA_Number val_B0, val_B1, val_B2, val_B3;
    ALPHA_Number val_A0, val_A1, val_A2, val_A3;

    alpha_setzero(sum0);
    alpha_setzero(sum1);
    alpha_setzero(sum2);
    alpha_setzero(sum3);

    // Loop over the column indices of B of the current row
    ALPHA_INT k = row_begin;
    for (; k < row_end - 1; k += 2) {
        col_B0 = csr_col_ind[k];
        col_B1 = csr_col_ind[k + 1];

        val_B0 = csr_val[k];
        val_B1 = csr_val[k + 1];

        val_A0 = A[col_B0 * lda + col];
        val_A1 = A[col_B1 * lda + col];

        alpha_madde(sum0, val_A0, val_B0);
        alpha_madde(sum1, val_A1, val_B1);
    }
    for (; k < row_end; ++k) {
        ALPHA_INT col_B    = csr_col_ind[k];
        ALPHA_Number val_B = csr_val[k];
        ALPHA_Number val_A = A[col_B * lda + col];

        // sum += val_A * val_B;
        alpha_madde(sum0, val_A, val_B);
    }
    alpha_add(sum0, sum0, sum1);

    // Write result back to matC
    // matC[row * ldc + col] = alpha * sum + beta * matC[row * ldc + col];
    ALPHA_Number tmp1, tmp2;
    alpha_mul(tmp1, alpha, sum0);
    alpha_mul(tmp2, beta, matC[row * ldc + col]);
    alpha_add(matC[row * ldc + col], tmp1, tmp2);
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
      const ALPHA_Number alpha,
      const ALPHA_Number *A,
      ALPHA_INT lda,
      const ALPHA_Number *csr_val,
      const ALPHA_INT *csr_row_ptr,
      const ALPHA_INT *csr_col_ind,
      const ALPHA_Number beta,
      ALPHA_Number *matC,
      ALPHA_INT ldc)
{
    const ALPHA_INT BLOCKSIZE = 512;

#ifdef S
    hipLaunchKernelGGL((gemmit_unroll2_kernel<BLOCKSIZE>), dim3((m - 1) / BLOCKSIZE + 1, n), dim3(BLOCKSIZE), 0, handle->stream, m, alpha, A, lda, csr_row_ptr, csr_col_ind, csr_val, beta, matC, ldc);
#else
    hipLaunchKernelGGL((gemmit_kernel<BLOCKSIZE>), dim3((m - 1) / BLOCKSIZE + 1, n), dim3(BLOCKSIZE), 0, handle->stream, m, alpha, A, lda, csr_row_ptr, csr_col_ind, csr_val, beta, matC, ldc);
#endif

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
