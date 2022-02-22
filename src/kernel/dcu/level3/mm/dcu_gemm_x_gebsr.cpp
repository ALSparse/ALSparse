#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/util_dcu.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

__global__ static void
gebsr_gemm_plain(alphasparse_layout_t dir,
                 ALPHA_INT mb,
                 ALPHA_INT n,
                 ALPHA_INT kb,
                 ALPHA_INT nnzb,
                 const ALPHA_Number alpha,
                 const ALPHA_Number *bsr_val,
                 const ALPHA_INT *bsr_row_ptr,
                 const ALPHA_INT *bsr_col_ind,
                 ALPHA_INT block_row_dim,
                 ALPHA_INT block_col_dim,
                 const ALPHA_Number *x,
                 ALPHA_INT ldx,
                 const ALPHA_Number beta,
                 ALPHA_Number *y,
                 ALPHA_INT ldy)
{
    ALPHA_INT tid    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ALPHA_INT stride = hipBlockDim_x * hipGridDim_x;

    ALPHA_INT m = mb * block_row_dim;

    for (ALPHA_INT j = tid * block_row_dim; j < n; j += stride * block_row_dim) {
        for (ALPHA_INT i = 0; i < m; ++i) {
            for (ALPHA_INT l = 0; l < block_row_dim; l++) {
                // y[index2(j, i, ldy)] *= beta;
                alpha_mul(y[index2(j + l, i, ldy)], beta, y[index2(j + l, i, ldy)]);
            }
        }
    }

    switch (dir) {
        case ALPHA_SPARSE_LAYOUT_ROW_MAJOR:
            for (ALPHA_INT c = tid * block_col_dim; c < n; c += block_col_dim * stride) { // choose a column from x
                for (ALPHA_INT r = 0; r < m; r += block_row_dim) { // choose a block of row
                    ALPHA_INT br = r / block_row_dim;
                    for (ALPHA_INT ai = bsr_row_ptr[br]; ai < bsr_row_ptr[br + 1]; ++ai) { // choose a block
                        const ALPHA_Number *blk = &bsr_val[ai * block_row_dim * block_col_dim];
                        for (ALPHA_INT cc = 0; cc < block_col_dim; ++cc)
                            for (ALPHA_INT lr = 0; lr < block_row_dim; ++lr) { // choose a inner row

                                ALPHA_INT ac = bsr_col_ind[ai] * block_col_dim;
                                ALPHA_Number extra;
                                alpha_setzero(extra);

                                for (ALPHA_INT lc = 0; lc < block_col_dim; ++lc) {
                                    // extra += blk[index2(lr, lc, bs)] * x[index2(c + cc, ac + lc, ldx)];
                                    alpha_madde(extra, blk[index2(lr, lc, block_col_dim)], x[index2(c + cc, ac + lc, ldx)]);
                                }
                                // y[index2(c + cc, r + lr, ldy)] += alpha * extra;
                                alpha_madde(y[index2(c + cc, r + lr, ldy)], alpha, extra);
                            }
                    }
                }
            }
            break;
        case ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR:
            for (ALPHA_INT c = tid * block_col_dim; c < n; c += block_col_dim * stride) { // choose a column from x
                for (ALPHA_INT r = 0; r < m; r += block_row_dim) { // choose a block of row
                    ALPHA_INT br = r / block_row_dim;
                    for (ALPHA_INT ai = bsr_row_ptr[br]; ai < bsr_row_ptr[br + 1]; ++ai) { // choose a block
                        for (ALPHA_INT cc = 0; cc < block_col_dim; ++cc)
                            for (ALPHA_INT lr = 0; lr < block_row_dim; ++lr) { // choose a inner row

                                ALPHA_INT ac            = bsr_col_ind[ai] * block_col_dim;
                                const ALPHA_Number *blk = &bsr_val[ai * block_col_dim * block_row_dim];
                                ALPHA_Number extra;
                                alpha_setzero(extra);

                                for (ALPHA_INT lc = 0; lc < block_col_dim; ++lc) {
                                    // extra += blk[index2(lc, lr, bs)] * x[index2(c + cc, ac + lc, ldx)];
                                    alpha_madde(extra, blk[index2(lc, lr, block_row_dim)], x[index2(c + cc, ac + lc, ldx)]);
                                }
                                // y[index2(c + cc, r + lr, ldy)] += alpha * extra;
                                alpha_madde(y[index2(c + cc, r + lr, ldy)], alpha, extra);
                            }
                    }
                }
            }
            break;
    }
}

static __global__ void gebsrmm_small(alphasparse_layout_t direction,
                                     ALPHA_INT Mb,
                                     ALPHA_INT N,
                                     ALPHA_Number alpha,
                                     const ALPHA_INT *__restrict__ bsr_row_ptr,
                                     const ALPHA_INT *__restrict__ bsr_col_ind,
                                     const ALPHA_Number *__restrict__ bsr_val,
                                     const ALPHA_Number *__restrict__ matB,
                                     ALPHA_INT ldb,
                                     ALPHA_Number beta,
                                     ALPHA_Number *__restrict__ matC,
                                     ALPHA_INT ldc)
{
    //TODO a general size for dim <= 4
    const ALPHA_INT ROW_BLOCK_DIM = 2;
    const ALPHA_INT COL_BLOCK_DIM = 4;
    const ALPHA_INT BLOCK_DIM     = 4;
    const ALPHA_INT BLK_SIZE_Y    = 16;

    const ALPHA_INT tidx       = hipThreadIdx_x;
    const ALPHA_INT tidy       = hipThreadIdx_y;
    const ALPHA_INT global_row = tidx + hipBlockIdx_x * ROW_BLOCK_DIM;
    const ALPHA_INT global_col = tidy + hipBlockIdx_y * BLK_SIZE_Y;
    const ALPHA_INT block_row  = hipBlockIdx_x;
    const ALPHA_INT colB       = global_col * ldb;
    const ALPHA_INT colC       = global_col * ldc;

    ALPHA_Number zero;
    alpha_setzero(zero);

    ALPHA_INT block_row_start = 0;
    ALPHA_INT block_row_end   = 0;
    if (block_row < Mb) {
        block_row_start = bsr_row_ptr[block_row];
        block_row_end   = bsr_row_ptr[block_row + 1];
    }

    __shared__ ALPHA_Number shared_B[BLOCK_DIM * BLK_SIZE_Y];
    __shared__ ALPHA_Number shared_A[BLOCK_DIM * BLOCK_DIM];

    ALPHA_Number sum = zero;

    const ALPHA_INT index                 = BLOCK_DIM * tidy + tidx;
    constexpr ALPHA_INT ROWXCOL_BLOCK_DIM = ROW_BLOCK_DIM * COL_BLOCK_DIM;
    const bool is_loading_B               = (global_col < N && tidx < COL_BLOCK_DIM);
    const bool is_loading_C               = (tidx < ROW_BLOCK_DIM && tidy < COL_BLOCK_DIM);

    for (ALPHA_INT k = block_row_start; k < block_row_end; k++) {
        ALPHA_INT block_col = (bsr_col_ind[k]);

        if (is_loading_B) {
            shared_B[index] = matB[colB + COL_BLOCK_DIM * block_col + tidx];
            // else
            // {
            //     shared_B[index] = matB[global_col + ldb * (COL_BLOCK_DIM * block_col + tidx)];
            // }
        } else {
            shared_B[index] = zero;
        }

        if (is_loading_C) {
            if (direction == ALPHA_SPARSE_LAYOUT_ROW_MAJOR) {
                shared_A[index] = bsr_val[ROWXCOL_BLOCK_DIM * k + COL_BLOCK_DIM * tidx + tidy];
            } else {
                shared_A[index] = bsr_val[ROWXCOL_BLOCK_DIM * k + ROW_BLOCK_DIM * tidy + tidx];
            }
        }

        __syncthreads();

        for (ALPHA_INT j = 0; j < COL_BLOCK_DIM; j++) {
            // sum += shared_A[ROW_BLOCK_DIM * j + tidx] * shared_B[ROW_BLOCK_DIM * tidy + j]
            alpha_madde(sum, shared_A[BLOCK_DIM * j + tidx], shared_B[BLOCK_DIM * tidy + j]);
        }

        __syncthreads();
    }

    if (block_row < Mb && global_col < N && tidx < ROW_BLOCK_DIM) {
        // matC[global_row + colC] = beta * matC[global_row + colC] + alpha * sum;
        ALPHA_Number t1, t2;
        alpha_mul(t1, beta, matC[global_row + colC]);
        alpha_mul(t2, alpha, sum);
        alpha_add(matC[global_row + colC], t1, t2);
    }
}

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      alphasparse_layout_t dir,
      ALPHA_INT mb,
      ALPHA_INT n,
      ALPHA_INT kb,
      ALPHA_INT nnzb,
      const ALPHA_Number alpha,
      const ALPHA_Number *bsr_val,
      const ALPHA_INT *bsr_row_ptr,
      const ALPHA_INT *bsr_col_ind,
      ALPHA_INT row_block_dim,
      ALPHA_INT col_block_dim,
      const ALPHA_Number *matB,
      ALPHA_INT ldb,
      const ALPHA_Number beta,
      ALPHA_Number *matC,
      ALPHA_INT ldc)
{
    if (row_block_dim == 2 && col_block_dim == 4) {
        ALPHA_INT M_         = row_block_dim;
        ALPHA_INT K_         = col_block_dim;
        ALPHA_INT BLOCK_DIM_ = 4;
        ALPHA_INT N_         = 16; //BLK_SIZE_Y
        dim3 gebsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / N_ + 1);
        dim3 gebsrmm_threads(BLOCK_DIM_, N_);
        hipLaunchKernelGGL(gebsrmm_small,
                           gebsrmm_blocks,
                           gebsrmm_threads,
                           0,
                           0,
                           dir,
                           mb,
                           n,
                           alpha,
                           bsr_row_ptr,
                           bsr_col_ind,
                           bsr_val,
                           matB,
                           ldb,
                           beta,
                           matC,
                           ldc);
    } else {
        const ALPHA_INT threadPerBlock = 256;
        const int blockPerGrid         = (threadPerBlock + n - 1) / threadPerBlock;

        hipLaunchKernelGGL(gebsr_gemm_plain, dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, dir, mb, n, kb, nnzb, alpha, bsr_val, bsr_row_ptr, bsr_col_ind, row_block_dim, col_block_dim, matB, ldb, beta, matC, ldc);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
