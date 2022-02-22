#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/common_dcu.h"

__global__ static void
bsr_gemv_plain(alphasparse_layout_t layout,
               ALPHA_INT mb,
               ALPHA_INT nb,
               ALPHA_INT nnzb,
               const ALPHA_Number alpha,
               const ALPHA_Number *bsr_val,
               const ALPHA_INT *bsr_row_ptr,
               const ALPHA_INT *bsr_col_ind,
               ALPHA_INT bs,
               const ALPHA_Number *x,
               const ALPHA_Number beta,
               ALPHA_Number *y)
{
    ALPHA_INT tid    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ALPHA_INT stride = hipBlockDim_x * hipGridDim_x;

    ALPHA_INT m_inner = mb;
    ALPHA_INT n_inner = nb;

    
    for (ALPHA_INT m = tid; m < m_inner; m += stride) {
        for (ALPHA_INT i = 0; i < bs; i++) {
            alpha_mul(y[m * bs + i], y[m * bs + i], beta);
            
        }
    }
    ALPHA_Number temp;
    alpha_setzero(temp);

    
    if (layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR) {
        for (ALPHA_INT i = tid; i < m_inner; i += stride) {
            for (ALPHA_INT ai = bsr_row_ptr[i]; ai < bsr_row_ptr[i + 1]; ai++) {
                
                for (ALPHA_INT row_inner = 0; row_inner < bs; row_inner++) {
                    for (ALPHA_INT col_inner = 0; col_inner < bs; col_inner++) {
                        alpha_mul(temp, alpha, bsr_val[ai * bs * bs + row_inner * bs + col_inner]);
                        alpha_mul(temp, temp, x[bs * bsr_col_ind[ai] + col_inner]);
                        alpha_add(y[bs * i + row_inner], y[bs * i + row_inner], temp);
                        
                    }
                    
                }
            }
        }
    }
    
    else if (layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR) {
        for (ALPHA_INT i = tid; i < m_inner; i += stride) {
            for (ALPHA_INT ai = bsr_row_ptr[i]; ai < bsr_row_ptr[i + 1]; ai++) {
                
                for (ALPHA_INT col_inner = 0; col_inner < bs; col_inner++) {
                    for (ALPHA_INT row_inner = 0; row_inner < bs; row_inner++) {
                        alpha_mul(temp, alpha, bsr_val[ai * bs * bs + col_inner * bs + row_inner]);
                        alpha_mul(temp, temp, x[bs * bsr_col_ind[ai] + col_inner]);
                        alpha_add(y[bs * i + row_inner], y[bs * i + row_inner], temp);
                        
                    }
                    
                }
            }
        }
    }
}


template <ALPHA_INT BLOCKSIZE, ALPHA_INT WFSIZE>
__global__ static void
    __launch_bounds__(BLOCKSIZE)
        bsrmvn_general_device(alphasparse_layout_t dir,
                              ALPHA_Number alpha,
                              const ALPHA_INT *__restrict__ bsr_row_ptr,
                              const ALPHA_INT *__restrict__ bsr_col_ind,
                              const ALPHA_Number *__restrict__ bsr_val,
                              ALPHA_INT bsr_dim,
                              const ALPHA_Number *__restrict__ x,
                              ALPHA_Number beta,
                              ALPHA_Number *__restrict__ y)
{
    
    ALPHA_INT lid = hipThreadIdx_x & (WFSIZE - 1);
    
    ALPHA_INT wid = hipThreadIdx_x / WFSIZE;
    
    ALPHA_INT nwf_per_block = hipBlockDim_x / WFSIZE;

    ALPHA_INT row = hipBlockIdx_x;

    
    ALPHA_INT row_begin = bsr_row_ptr[row];
    ALPHA_INT row_end   = bsr_row_ptr[row + 1];

    for (ALPHA_INT bi = wid; bi < bsr_dim; bi += nwf_per_block) {
        
        ALPHA_Number sum;
        alpha_setzero(sum);

        for (ALPHA_INT j = row_begin; j < row_end; ++j) {
            
            ALPHA_INT col = bsr_col_ind[j];
            
            for (ALPHA_INT bj = lid; bj < bsr_dim; bj += WFSIZE) {
                
                alpha_madde(sum, bsr_val[BSR_IND(j, bi, bj, dir)], x[bsr_dim * col + bj]);
            }
        }

        sum = wfreduce_sum<WFSIZE>(sum);

        if (lid == WFSIZE - 1) {
            
            ALPHA_Number tmp1, tmp2;
            alpha_mul(tmp1, beta, y[row * bsr_dim + bi]);
            alpha_mul(tmp2, alpha, sum);
            alpha_add(y[row * bsr_dim + bi], tmp1, tmp2);
        }
    }
}


#ifdef __cplusplus
extern "C" {
#endif
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
      ALPHA_INT bsr_dim,
      const ALPHA_Number *x,
      const ALPHA_Number beta,
      ALPHA_Number *y)
{
    
    const ALPHA_INT blocks_per_row = nnzb / mb;
    
    {
        if (bsr_dim <= 8) {
            hipLaunchKernelGGL((bsrmvn_general_device<64, 8>), dim3(mb), dim3(8 * 8), 0, handle->stream, dir, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, bsr_dim, x, beta, y);
        } else if (bsr_dim <= 16) {
            hipLaunchKernelGGL((bsrmvn_general_device<256, 16>), dim3(mb), dim3(16 * 16), 0, handle->stream, dir, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, bsr_dim, x, beta, y);
        } else {
            hipLaunchKernelGGL((bsrmvn_general_device<1024, 32>), dim3(mb), dim3(32 * 32), 0, handle->stream, dir, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, bsr_dim, x, beta, y);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif
