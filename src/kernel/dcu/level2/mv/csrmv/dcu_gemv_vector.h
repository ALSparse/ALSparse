#pragma once

#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/common_dcu.h"

#include "dcu_gemv_common.h"


template <ALPHA_INT BLOCKSIZE, ALPHA_INT WF_SIZE>
__launch_bounds__(BLOCKSIZE)
    __global__ static void csr_gemv_vector(ALPHA_INT m,
                                           ALPHA_Number alpha,
                                           const ALPHA_INT *row_offset,
                                           const ALPHA_INT *csr_col_ind,
                                           const ALPHA_Number *csr_val,
                                           const ALPHA_Number *x,
                                           ALPHA_Number beta,
                                           ALPHA_Number *y,
                                           u_int32_t flag)
{
    const ALPHA_INT lid               = hipThreadIdx_x & (WF_SIZE - 1); 
    const ALPHA_INT VECTORS_PER_BLOCK = BLOCKSIZE / WF_SIZE; 
    const ALPHA_INT vector_lane       = threadIdx.x / WF_SIZE; 

    __shared__ volatile ALPHA_INT ptrs[VECTORS_PER_BLOCK][2];

    ALPHA_INT gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
    ALPHA_INT nwf = hipGridDim_x * BLOCKSIZE / WF_SIZE;
    
    for (ALPHA_INT row = gid / WF_SIZE; row < m; row += nwf) {
        ALPHA_INT row_start, row_end;
        
        row_start = row_offset[row];
        row_end   = row_offset[row + 1];
        ALPHA_Number sum;
        alpha_setzero(sum);
        
        for (ALPHA_INT j = row_start + lid; j < row_end; j += WF_SIZE) {
            
            alpha_madde(sum, csr_val[j], x[csr_col_ind[j]]);
        }

        sum = wfreduce_sum<WF_SIZE>(sum);
        
        if (lid == WF_SIZE - 1) {
            ALPHA_Number t1, t2;
            alpha_mul(t1, y[row], beta);
            alpha_mul(t2, sum, alpha);
            alpha_add(y[row], t1, t2);
        }
    }
}

alphasparse_status_t csr_gemv_vector_dispatch(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              ALPHA_INT nnz,
                                              const ALPHA_Number alpha,
                                              const ALPHA_Number *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              const ALPHA_Number *x,
                                              const ALPHA_Number beta,
                                              ALPHA_Number *y,
                                              u_int32_t flag)
{
    const ALPHA_INT CSRMVN_DIM  = 512;
    const ALPHA_INT nnz_per_row = nnz / m;

    const ALPHA_INT block_num_base = (m - 1) / CSRMVN_DIM + 1;

    if (nnz_per_row < 4) {
        dim3 csrmvn_blocks(block_num_base * 2);
        dim3 csrmvn_threads(CSRMVN_DIM);
        hipLaunchKernelGGL((csr_gemv_vector<CSRMVN_DIM, 2>),
                           csrmvn_blocks,
                           csrmvn_threads,
                           0,
                           handle->stream,
                           m,
                           alpha,
                           csr_row_ptr,
                           csr_col_ind,
                           csr_val,
                           x,
                           beta,
                           y,
                           flag);
    } else if (nnz_per_row < 8) {
        dim3 csrmvn_blocks(block_num_base * 4);
        dim3 csrmvn_threads(CSRMVN_DIM);
        hipLaunchKernelGGL((csr_gemv_vector<CSRMVN_DIM, 4>),
                           csrmvn_blocks,
                           csrmvn_threads,
                           0,
                           handle->stream,
                           m,
                           alpha,
                           csr_row_ptr,
                           csr_col_ind,
                           csr_val,
                           x,
                           beta,
                           y,
                           flag);
    } else if (nnz_per_row < 16) {
        dim3 csrmvn_blocks(block_num_base * 8);
        dim3 csrmvn_threads(CSRMVN_DIM);
        hipLaunchKernelGGL((csr_gemv_vector<CSRMVN_DIM, 8>),
                           csrmvn_blocks,
                           csrmvn_threads,
                           0,
                           handle->stream,
                           m,
                           alpha,
                           csr_row_ptr,
                           csr_col_ind,
                           csr_val,
                           x,
                           beta,
                           y,
                           flag);
    } else if (nnz_per_row < 32) {
        dim3 csrmvn_blocks(block_num_base * 16);
        dim3 csrmvn_threads(CSRMVN_DIM);
        hipLaunchKernelGGL((csr_gemv_vector<CSRMVN_DIM, 16>),
                           csrmvn_blocks,
                           csrmvn_threads,
                           0,
                           handle->stream,
                           m,
                           alpha,
                           csr_row_ptr,
                           csr_col_ind,
                           csr_val,
                           x,
                           beta,
                           y,
                           flag);
    } else if (nnz_per_row < 64) {
        dim3 csrmvn_blocks((block_num_base * 32));
        dim3 csrmvn_threads(CSRMVN_DIM);
        hipLaunchKernelGGL((csr_gemv_vector<CSRMVN_DIM, 32>),
                           csrmvn_blocks,
                           csrmvn_threads,
                           0,
                           handle->stream,
                           m,
                           alpha,
                           csr_row_ptr,
                           csr_col_ind,
                           csr_val,
                           x,
                           beta,
                           y,
                           flag);
    } else {
        dim3 csrmvn_blocks(block_num_base * 64);
        dim3 csrmvn_threads(CSRMVN_DIM);
        hipLaunchKernelGGL((csr_gemv_vector<CSRMVN_DIM, 64>),
                           csrmvn_blocks,
                           csrmvn_threads,
                           0,
                           handle->stream,
                           m,
                           alpha,
                           csr_row_ptr,
                           csr_col_ind,
                           csr_val,
                           x,
                           beta,
                           y,
                           flag);
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
