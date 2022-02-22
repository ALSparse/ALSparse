#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/common_dcu.h"

__global__ static void
gebsr_gemv_plain(alphasparse_layout_t layout,
                 ALPHA_INT mb,
                 ALPHA_INT nb,
                 ALPHA_INT nnzb,
                 const ALPHA_Number alpha,
                 const ALPHA_Number *bsr_val,
                 const ALPHA_INT *bsr_row_ptr,
                 const ALPHA_INT *bsr_col_ind,
                 ALPHA_INT row_block_dim,
                 ALPHA_INT col_block_dim,
                 const ALPHA_Number *x,
                 const ALPHA_Number beta,
                 ALPHA_Number *y)
{
    ALPHA_INT tid    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ALPHA_INT stride = hipBlockDim_x * hipGridDim_x;

    ALPHA_INT m_inner = mb;
    ALPHA_INT n_inner = nb;

    // y = y * beta
    for (ALPHA_INT m = tid; m < m_inner; m += stride) {
        for (ALPHA_INT i = 0; i < row_block_dim; i++) {
            alpha_mul(y[m * row_block_dim + i], y[m * row_block_dim + i], beta);
            //y[m] *= beta;
        }
    }
    ALPHA_Number temp;
    alpha_setzero(temp);

    // For matC, block_layout is defaulted as row_major
    if (layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR) {
        for (ALPHA_INT i = tid; i < m_inner; i += stride) {
            for (ALPHA_INT ai = bsr_row_ptr[i]; ai < bsr_row_ptr[i + 1]; ai++) {
                // block[ai]: [i][bsr_col_ind[ai]]
                for (ALPHA_INT row_inner = 0; row_inner < row_block_dim; row_inner++) {
                    for (ALPHA_INT col_inner = 0; col_inner < col_block_dim; col_inner++) {
                        alpha_mul(temp, alpha, bsr_val[ai * row_block_dim * col_block_dim + row_inner * row_block_dim + col_inner]);
                        alpha_mul(temp, temp, x[col_block_dim * bsr_col_ind[ai] + col_inner]);
                        alpha_add(y[row_block_dim * i + row_inner], y[row_block_dim * i + row_inner], temp);
                        //y[bs*i+row_inner] += alpha*bsr_val[ai*bs*bs+row_inner*bs+col_inner]*x[bs*bsr_col_ind[ai]+col_inner];
                    }
                    // over for block
                }
            }
        }
    }
    // For Fortran, block_layout is defaulted as col_major
    else if (layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR) {
        for (ALPHA_INT i = tid; i < m_inner; i += stride) {
            for (ALPHA_INT ai = bsr_row_ptr[i]; ai < bsr_row_ptr[i + 1]; ai++) {
                // block[ai]: [i][A->cols[ai]]
                for (ALPHA_INT col_inner = 0; col_inner < col_block_dim; col_inner++) {
                    for (ALPHA_INT row_inner = 0; row_inner < row_block_dim; row_inner++) {
                        alpha_mul(temp, alpha, bsr_val[ai * row_block_dim * col_block_dim + col_inner * col_block_dim + row_inner]);
                        alpha_mul(temp, temp, x[col_block_dim * bsr_col_ind[ai] + col_inner]);
                        alpha_add(y[row_block_dim * i + row_inner], y[row_block_dim * i + row_inner], temp);
                        //y[bs*i+row_inner] += alpha*bsr_val[ai*bs*bs+col_inner*bs+row_inner]*x[bs*bsr_col_ind[ai]+col_inner];
                    }
                    // over for block
                }
            }
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      alphasparse_layout_t dir,
      ALPHA_INT mb,
      ALPHA_INT nb,
      ALPHA_INT nnzb,
      const ALPHA_Number alpha,
      const ALPHA_Number *bsr_val,
      const ALPHA_INT *bsr_row_ptr,
      const ALPHA_INT *bsr_col_ind,
      ALPHA_INT row_block_dim,
      ALPHA_INT col_block_dim,
      const ALPHA_Number *x,
      const ALPHA_Number beta,
      ALPHA_Number *y)
{
    const ALPHA_INT threadPerBlock = 256;
    const int blockPerGrid = min(32, (threadPerBlock + mb - 1) / threadPerBlock);

    hipLaunchKernelGGL(gebsr_gemv_plain, blockPerGrid, threadPerBlock, 0, handle->stream,
                       dir, mb, nb, nnzb, alpha, bsr_val, bsr_row_ptr, bsr_col_ind, row_block_dim, col_block_dim, x, beta, y);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
#undef BLOCKSIZE
#undef WFSIZE

#ifdef __cplusplus
}
#endif /*__cplusplus */
