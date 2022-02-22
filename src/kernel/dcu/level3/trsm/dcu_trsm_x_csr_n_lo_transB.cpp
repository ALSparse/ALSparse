#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/util_dcu.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

__global__ static void
get_diags(ALPHA_INT m,
          const ALPHA_Number *csr_val,
          const ALPHA_INT *csr_row_ptr,
          const ALPHA_INT *csr_col_ind,
          ALPHA_Number *diag)
{
    const ALPHA_INT global_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const ALPHA_INT stride    = hipBlockDim_x * hipGridDim_x;
    for (ALPHA_INT r = global_id; r < m; r += stride) {
        for (ALPHA_INT ai = csr_row_ptr[r]; ai < csr_row_ptr[r + 1]; ai++) {
            ALPHA_INT ac = csr_col_ind[ai];
            if (ac == r) {
                diag[r] = csr_val[ai];
            }
        }
    }
}

__global__ static void trsm_plain(ALPHA_INT m,
                                  ALPHA_INT nrhs,
                                  ALPHA_INT nnz,
                                  const ALPHA_Number alpha,
                                  const ALPHA_Number *csr_val,
                                  const ALPHA_INT *csr_row_ptr,
                                  const ALPHA_INT *csr_col_ind,
                                  ALPHA_Number *y,
                                  ALPHA_INT ldb_,
                                  ALPHA_Number *diag)
{
    const ALPHA_INT global_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const ALPHA_INT stride    = hipBlockDim_x * hipGridDim_x;

    ALPHA_INT ldb = nrhs; //transB make the ldb_ useless
    for (ALPHA_INT cc = global_id; cc < nrhs; cc += stride) {
        // for (ALPHA_INT r = 0; r < m; r++)
        // {
        //     for (ALPHA_INT ai = csr_row_ptr[r]; ai < csr_row_ptr[r + 1]; ai++)
        //     {
        //         ALPHA_INT ac = csr_col_ind[ai];
        //         if (ac == r)
        //         {
        //             diag[r] = csr_val[ai];
        //         }
        //     }
        // }
        for (ALPHA_INT r = 0; r < m; r++) {
            ALPHA_Number temp;
            alpha_setzero(temp);
            for (ALPHA_INT ai = csr_row_ptr[r]; ai < csr_row_ptr[r + 1]; ai++) {
                ALPHA_INT ac = csr_col_ind[ai];
                if (ac < r) {
                    alpha_madde(temp, csr_val[ai], y[index2(ac, cc, ldb)]);
                }
            }
            ALPHA_Number t;
            alpha_mul(t, alpha, y[index2(r, cc, ldb)]);
            alpha_sube(t, temp);
            alpha_div(y[index2(r, cc, ldb)], t, diag[r]);
            // y[r] = (alpha * x[r] - temp) / diag[r];
        }
    }
}

template <ALPHA_INT BLOCKSIZE>
__global__ static void trsm_opt_SM(const ALPHA_INT *csrRowPtr,
                                   const ALPHA_INT *csrColIdx,
                                   const ALPHA_Number *csrVal,
                                   const ALPHA_INT m,
                                   const ALPHA_INT nrhs,
                                   const ALPHA_INT nnz,
                                   const ALPHA_Number alpha,
                                   const ALPHA_Number *diag,
                                   ALPHA_Number *y,
                                   ALPHA_INT ldb,
                                   ALPHA_INT *get_value)

{
    const ALPHA_INT stride    = hipBlockDim_x * hipGridDim_x;
    const ALPHA_INT cidx      = hipBlockIdx_x;
    const ALPHA_INT thread_id = hipThreadIdx_x + cidx * hipBlockDim_x;

    ALPHA_INT col, j;
    // ALPHA_INT blockPern = (nrhs - 1) / hipBlockDim_x + 1;
    ALPHA_INT row = cidx % m;

    ALPHA_Number left_sum;
    alpha_setzero(left_sum);

    __shared__ ALPHA_INT scsr_col_ind[BLOCKSIZE];
    __shared__ ALPHA_Number scsr_val[BLOCKSIZE];

    ALPHA_INT row_begin = csrRowPtr[row];
    ALPHA_INT row_end   = csrRowPtr[row + 1];

    // Column index into B
    ALPHA_INT col_B = hipBlockIdx_x / m * BLOCKSIZE + hipThreadIdx_x;

    // Index into B (i,j)
    // ALPHA_INT idx_B = col_B * ldb + row;
    ALPHA_INT idx_B = row * ldb + col_B; //row major

    // Index into done array
    ALPHA_INT id = hipBlockIdx_x / m * m;

    // Initialize local sum with alpha and X
    ALPHA_Number local_sum;

    if (col_B < nrhs) {
        alpha_mul(local_sum, alpha, y[idx_B]);
    } else {
        alpha_setzero(local_sum);
    }

    for (ALPHA_INT j = row_begin; j < row_end; ++j) {
        // Project j onto [0, BLOCKSIZE-1]
        ALPHA_INT k = (j - row_begin) & (BLOCKSIZE - 1);

        // Preload column indices and values into shared memory
        // This happens only once for each chunk of BLOCKSIZE elements
        if (k == 0) {
            if (hipThreadIdx_x < row_end - j) {
                scsr_col_ind[hipThreadIdx_x] = csrColIdx[hipThreadIdx_x + j];
                scsr_val[hipThreadIdx_x]     = csrVal[hipThreadIdx_x + j];
            } else {
                scsr_col_ind[hipThreadIdx_x] = -1;
                alpha_setzero(scsr_val[hipThreadIdx_x]);
                // alpha_sube(scsr_val[hipThreadIdx_x], 1);
            }
        }

        // Wait for preload to finish
        __syncthreads();

        // Current column this lane operates on
        ALPHA_INT local_col = scsr_col_ind[k];

        if (local_col > row) {
            break;
        }

        // Local value this lane operates with
        ALPHA_Number local_val = scsr_val[k];

        if (local_col == row) {
            break;
        }

        // Spin loop until dependency has been resolved
        if (hipThreadIdx_x == 0) {
            int local_done             = atomicOr(&get_value[local_col + id], 0);
            unsigned int times_through = 0;
            while (!local_done) {
                local_done = atomicOr(&get_value[local_col + id], 0);
            }
        }

        // Wait for spin looping thread to finish as the whole block depends on this row
        __syncthreads();

        // Make sure updated B is visible globally
        __threadfence();

        // Index into X
        // ALPHA_INT idx_X = col_B * ldb + local_col;
        ALPHA_INT idx_X = local_col * ldb + col_B; //row major

        // Local sum computation for each lane
        if (col_B < nrhs) {
            alpha_msube(local_sum, local_val, y[idx_X]);
        } else {
            alpha_setzero(local_sum);
        }
    }

    // Store result in B
    if (col_B < nrhs) {
        alpha_div(y[idx_B], local_sum, diag[row]);
    }

    // Wait for all threads to finish writing into global memory before we mark the row "done"
    __syncthreads();

    // Make sure B is written to global memory before setting row is done flag
    __threadfence();

    if (hipThreadIdx_x == 0) {
        // Write the "row is done" flag
        atomicOr(&get_value[row + id], 1);
    }
}

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

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
    ALPHA_Number *diag;
    hipMalloc((void **)&diag, sizeof(ALPHA_Number) * m);
    hipMemset(diag, '\0', sizeof(ALPHA_Number) * m);

    const ALPHA_INT threadPerBlock = 256;
    const int blockPerGrid         = min(32, (threadPerBlock + nrhs - 1) / threadPerBlock);

    //todo diag的计算是否应合并
    hipLaunchKernelGGL(get_diags, dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, m, csr_val, csr_row_ptr, csr_col_ind, diag);

    // hipLaunchKernelGGL(trsm_plain, dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream,
    //                    m, nrhs, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, B, ldb, diag);

    const ALPHA_INT blockdim = 1024;
    const ALPHA_INT griddim  = ((nrhs - 1) / blockdim + 1) * m;
    const ALPHA_INT narrays  = (nrhs - 1) / blockdim + 1;
    ALPHA_INT *get_value;

    hipMalloc((void **)&get_value, sizeof(ALPHA_INT) * m * narrays);

    hipMemsetAsync(get_value, 0, sizeof(ALPHA_INT) * m * narrays, handle->stream);

    hipLaunchKernelGGL((trsm_opt_SM<blockdim>), dim3(griddim), dim3(blockdim), 0, handle->stream, csr_row_ptr, csr_col_ind, csr_val, m, nrhs, nnz, alpha, diag, B, ldb, get_value);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
