#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/util_dcu.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

template <unsigned int DIM_X, unsigned int DIM_Y, typename T>
__launch_bounds__(DIM_X *DIM_Y) __global__ void csrsm_transpose(ALPHA_INT m,
                                                                ALPHA_INT n,
                                                                const T *__restrict__ A,
                                                                ALPHA_INT lda,
                                                                T *__restrict__ B,
                                                                ALPHA_INT ldb)
{
    ALPHA_INT lid = hipThreadIdx_x & (DIM_X - 1);
    ALPHA_INT wid = hipThreadIdx_x / DIM_X;

    ALPHA_INT row_A = hipBlockIdx_x * DIM_X + lid;
    ALPHA_INT row_B = hipBlockIdx_x * DIM_X + wid;

    __shared__ T sdata[DIM_X][DIM_X];

    for (int j = 0; j < n; j += DIM_X) {
        __syncthreads();

        int col_A = j + wid;

        for (int k = 0; k < DIM_X; k += DIM_Y) {
            if (row_A < m && col_A + k < n) {
                sdata[wid + k][lid] = A[row_A + lda * (col_A + k)];
            }
        }

        __syncthreads();

        int col_B = j + lid;

        for (int k = 0; k < DIM_X; k += DIM_Y) {
            if (col_B < n && row_B + k < m) {
                B[col_B + ldb * (row_B + k)] = sdata[lid][wid + k];
            }
        }
    }
}

template <unsigned int DIM_X, unsigned int DIM_Y, typename T>
__launch_bounds__(DIM_X *DIM_Y) __global__ void csrsm_transpose_back(ALPHA_INT m,
                                                                     ALPHA_INT n,
                                                                     const T *__restrict__ A,
                                                                     ALPHA_INT lda,
                                                                     T *__restrict__ B,
                                                                     ALPHA_INT ldb)
{
    ALPHA_INT lid = hipThreadIdx_x & (DIM_X - 1);
    ALPHA_INT wid = hipThreadIdx_x / DIM_X;

    ALPHA_INT row_A = hipBlockIdx_x * DIM_X + wid;
    ALPHA_INT row_B = hipBlockIdx_x * DIM_X + lid;

    __shared__ T sdata[DIM_X][DIM_X];

    for (int j = 0; j < n; j += DIM_X) {
        __syncthreads();

        int col_A = j + lid;

        for (int k = 0; k < DIM_X; k += DIM_Y) {
            if (col_A < n && row_A + k < m) {
                sdata[wid + k][lid] = A[col_A + lda * (row_A + k)];
            }
        }

        __syncthreads();

        int col_B = j + wid;

        for (int k = 0; k < DIM_X; k += DIM_Y) {
            if (row_B < m && col_B + k < n) {
                B[row_B + ldb * (col_B + k)] = sdata[lid][wid + k];
            }
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define SM_SIZE 1024

__global__ static void trsm_plain(ALPHA_INT m,
                                  ALPHA_INT nrhs,
                                  ALPHA_INT nnz,
                                  const ALPHA_Number alpha,
                                  const ALPHA_Number *csr_val,
                                  const ALPHA_INT *csr_row_ptr,
                                  const ALPHA_INT *csr_col_ind,
                                  ALPHA_Number *y,
                                  ALPHA_INT ldb)
{
    const ALPHA_INT global_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const ALPHA_INT stride    = hipBlockDim_x * hipGridDim_x;

    for (ALPHA_INT cc = global_id; cc < nrhs; cc += stride) {
        for (ALPHA_INT r = 0; r < m; r++) {
            ALPHA_Number temp;
            alpha_setzero(temp);
            for (ALPHA_INT ai = csr_row_ptr[r]; ai < csr_row_ptr[r + 1]; ai++) {
                ALPHA_INT ac = csr_col_ind[ai];
                if (ac < r) {
                    alpha_madde(temp, csr_val[ai], y[index2(cc, ac, ldb)]);
                }
            }
            ALPHA_Number t;
            alpha_mul(t, alpha, y[index2(cc, r, ldb)]);
            alpha_sub(y[index2(cc, r, ldb)], t, temp);
            // y[r] = (alpha * x[r] - temp);
        }
    }
}

#define BLOCKSIZE 1024
__global__ static void trsm_reduce_sum(ALPHA_INT m,
                                       ALPHA_INT nrhs,
                                       ALPHA_INT nnz,
                                       const ALPHA_Number alpha,
                                       const ALPHA_Number *csr_val,
                                       const ALPHA_INT *csr_row_ptr,
                                       const ALPHA_INT *csr_col_ind,
                                       ALPHA_Number *y,
                                       ALPHA_INT ldb)
{
    const ALPHA_INT global_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const ALPHA_INT stride    = hipBlockDim_x * hipGridDim_x;

    __shared__ ALPHA_Number cache[BLOCKSIZE];

    ALPHA_INT cc = hipBlockIdx_x;
    {
        for (ALPHA_INT r = 0; r < m; r++) {
            int cacheidx = threadIdx.x;

            // alpha_setzero(cache[cacheidx]);
            // __syncthreads();

            ALPHA_Number temp;
            alpha_setzero(temp);
            ALPHA_INT csr_row_start = csr_row_ptr[r];
            ALPHA_INT csr_row_end   = csr_row_ptr[r + 1];
            for (ALPHA_INT ai = csr_row_start + hipThreadIdx_x; ai < csr_row_end; ai += BLOCKSIZE) {
                ALPHA_INT ac = csr_col_ind[ai];
                if (ac < r) {
                    alpha_madde(temp, csr_val[ai], y[index2(cc, ac, ldb)]);
                }
            }
            cache[cacheidx] = temp;
            __syncthreads();

            //规约
            ALPHA_INT i = BLOCKSIZE / 2;
            while (i != 0) {
                if (cacheidx < i) // 只需要线程号小于i的线程参与计算
                {
                    // cache[cacheidx] += cache[cacheidx + i]; // 两两求和
                    alpha_add(cache[cacheidx], cache[cacheidx], cache[cacheidx + i]);
                }
                i /= 2; // 循环变量
                __syncthreads();
            }

            if (hipThreadIdx_x == 0) {
                ALPHA_Number t;
                alpha_mul(t, alpha, y[index2(cc, r, ldb)]);
                alpha_sub(y[index2(cc, r, ldb)], t, cache[0]);
                // y[r] = (alpha * x[r] - temp);
            }
        }
    }
}

__global__ static void spts_syncfree_cuda_executor_csr_wrt_thread(const ALPHA_INT *csrRowPtr,
                                                                  const ALPHA_INT *csrColIdx,
                                                                  const ALPHA_Number *csrVal,
                                                                  const ALPHA_INT m,
                                                                  const ALPHA_INT nrhs,
                                                                  const ALPHA_INT nnz,
                                                                  const ALPHA_Number alpha,
                                                                  ALPHA_Number *y,
                                                                  ALPHA_INT ldb)

{
    const ALPHA_INT thread_id = hipThreadIdx_x;
    const ALPHA_INT stride    = hipBlockDim_x;
    const ALPHA_INT cidx      = hipBlockIdx_x;
    extern __shared__ ALPHA_INT get_value[];
    for (ALPHA_INT b = cidx; b < nrhs; b += hipGridDim_x) {
        for (ALPHA_INT i = thread_id; i < m; i += stride) {
            get_value[i] = 0;
        }
        __syncthreads();

        ALPHA_INT col, j;
        ALPHA_Number yi;
        ALPHA_Number left_sum;
        alpha_setzero(left_sum);

        for (ALPHA_INT i = thread_id; i < m; i += stride) {
            alpha_setzero(left_sum);
            j = csrRowPtr[i];
            while (j < csrRowPtr[i + 1]) {
                col = csrColIdx[j];

                while (get_value[col] == 1) {
                    if (col < i) {
                        alpha_madde(left_sum, csrVal[j], y[ldb * b + col]);
                    } else
                        break;
                    j++;
                    col = csrColIdx[j];
                }

                ALPHA_INT tmp_try = !(i ^ col);
                {
                    ALPHA_Number tmp;
                    alpha_mul(tmp, alpha, y[ldb * b + i]);
                    alpha_sub(yi, tmp, left_sum);
                    alpha_cross_entropy(y[ldb * b + i], yi, y[ldb * b + i], tmp_try);
                    __threadfence();
                    // __builtin_amdgcn_s_sleep(1);
                    get_value[i] = tmp_try | get_value[i];
                    __threadfence();

                    if (tmp_try)
                        break;
                }
            }
        }
    }
}

__global__ static void trsm_opt_SM(const ALPHA_INT *csrRowPtr,
                                   const ALPHA_INT *csrColIdx,
                                   const ALPHA_Number *csrVal,
                                   const ALPHA_INT m,
                                   const ALPHA_INT nrhs,
                                   const ALPHA_INT nnz,
                                   const ALPHA_Number alpha,
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
        y[idx_B] = local_sum;
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
    ALPHA_INT blockdim = BLOCKSIZE;
    // int blockdim = 512;
    // while(nrhs <= blockdim && blockdim > 32)
    // {
    //     blockdim >>= 1;
    // }
    // blockdim <<= 1;

    // Leading dimension for transposed B
    ALPHA_INT ldimB  = nrhs;
    ALPHA_Number *Bt = NULL;
    hipMalloc((void **)&Bt, sizeof(ALPHA_Number) * m * ldimB);

    {
#define CSRSM_DIM_X 32
#define CSRSM_DIM_Y 8
        dim3 csrsm_blocks((m - 1) / CSRSM_DIM_X + 1);
        dim3 csrsm_threads(CSRSM_DIM_X * CSRSM_DIM_Y);

        hipLaunchKernelGGL((csrsm_transpose<CSRSM_DIM_X, CSRSM_DIM_Y, ALPHA_Number>),
                           csrsm_blocks,
                           csrsm_threads,
                           0,
                           handle->stream,
                           m,
                           nrhs,
                           B,
                           ldb,
                           Bt,
                           ldimB);

#undef CSRSM_DIM_X
#undef CSRSM_DIM_Y
    }

    ALPHA_INT griddim = ((nrhs - 1) / blockdim + 1) * m;
    ALPHA_INT narrays = (nrhs - 1) / blockdim + 1;
    ALPHA_INT *get_value;

    hipMalloc((void **)&get_value, sizeof(ALPHA_INT) * m * narrays);

    hipMemsetAsync(get_value, 0, sizeof(ALPHA_INT) * m * narrays, handle->stream);

    // hipLaunchKernelGGL(trsm_opt_SM, dim3(griddim), dim3(blockdim), 0, handle->stream,
    //                    csr_row_ptr, csr_col_ind, csr_val, m, nrhs, nnz, alpha, B, ldb, get_value);

    hipLaunchKernelGGL(trsm_opt_SM, dim3(griddim), dim3(blockdim), 0, handle->stream, csr_row_ptr, csr_col_ind, csr_val, m, nrhs, nnz, alpha, Bt, ldimB, get_value);

    {
#define CSRSM_DIM_X 32
#define CSRSM_DIM_Y 8
        dim3 csrsm_blocks((m - 1) / CSRSM_DIM_X + 1);
        dim3 csrsm_threads(CSRSM_DIM_X * CSRSM_DIM_Y);

        hipLaunchKernelGGL((csrsm_transpose_back<CSRSM_DIM_X, CSRSM_DIM_Y, ALPHA_Number>),
                           csrsm_blocks,
                           csrsm_threads,
                           0,
                           handle->stream,
                           m,
                           nrhs,
                           Bt,
                           ldimB,
                           B,
                           ldb);
#undef CSRSM_DIM_X
#undef CSRSM_DIM_Y
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
#undef BLOCKSIZE

#ifdef __cplusplus
}
#endif /*__cplusplus */
