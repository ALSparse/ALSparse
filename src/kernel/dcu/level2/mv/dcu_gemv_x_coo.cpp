#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

#define WF_SIZE   64
#define BLOCKSIZE 128

__launch_bounds__(BLOCKSIZE) static __global__ void coomvn_general_wf_reduce(ALPHA_INT nnz,
                                                                             ALPHA_INT loops,
                                                                             ALPHA_Number alpha,
                                                                             const ALPHA_INT *__restrict__ coo_row_ind,
                                                                             const ALPHA_INT *__restrict__ coo_col_ind,
                                                                             const ALPHA_Number *__restrict__ coo_val,
                                                                             const ALPHA_Number *__restrict__ x,
                                                                             ALPHA_Number *__restrict__ y,
                                                                             ALPHA_INT *__restrict__ row_block_red,
                                                                             ALPHA_Number *__restrict__ val_block_red)
{
    ALPHA_INT tid = hipThreadIdx_x;
    ALPHA_INT gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
    ALPHA_INT lid = tid & (WF_SIZE - 1); 
    
    ALPHA_INT wid = gid / WF_SIZE;

    if (lid == 0) {
        *(row_block_red + wid) = -1;
        alpha_setzero(*(val_block_red + wid));
    }

    ALPHA_INT offset = wid * loops * WF_SIZE;

    __shared__ ALPHA_INT shared_row[BLOCKSIZE];
    __shared__ ALPHA_Number shared_val[BLOCKSIZE];

    shared_row[tid] = -1;
    alpha_setzero(*(val_block_red + wid));

    __syncthreads();

    if (offset + lid >= nnz) {
        return;
    }

    ALPHA_INT row;
    ALPHA_Number val;

    ALPHA_INT idx = offset + lid;

    while (idx < offset + loops * WF_SIZE) {
        if (idx < nnz) {
            row            = *(coo_row_ind + idx);
            ALPHA_Number v = *(coo_val + idx);
            alpha_mul(val, alpha, v);
            ALPHA_Number tmp = *(x + *(coo_col_ind + idx));
            alpha_mul(val, val, tmp);
        } else {
            row = -1;
            alpha_setzero(val);
        }
        
        if (idx > offset && lid == 0) {
            ALPHA_INT prevrow = shared_row[tid + WF_SIZE - 1];
            if (row == prevrow) {
                alpha_add(val, val, shared_val[tid + WF_SIZE - 1]);
            } else if (prevrow >= 0) {
                alpha_add(y[prevrow], y[prevrow], shared_val[tid + WF_SIZE - 1]);
            }
        }

        __syncthreads();

        shared_row[tid] = row;
        shared_val[tid] = val;

        __syncthreads();

        for (ALPHA_INT j = 1; j < WF_SIZE; j <<= 1) {
            if (lid >= j) {
                if (row == shared_row[tid - j]) {
                    alpha_add(val, val, shared_val[tid - j]);
                }
            }
            __syncthreads();
            shared_val[tid] = val;
            __syncthreads();
        }

        if (lid < WF_SIZE - 1) {
            if (row != shared_row[tid + 1] && row >= 0) {
                alpha_add(y[row], y[row], val);
            }
        }
        idx += WF_SIZE;
    }

    if (lid == WF_SIZE - 1) {
        *(row_block_red + wid) = row;
        *(val_block_red + wid) = val;
    }
}


static __device__ void segmented_blockreduce(const ALPHA_INT *__restrict__ rows, ALPHA_Number *__restrict__ vals)
{
    ALPHA_INT tid = hipThreadIdx_x;

    
    for (ALPHA_INT j = 1; j < BLOCKSIZE; j <<= 1) {
        ALPHA_Number val;
        alpha_setzero(val);
        if (tid >= j) {
            if (rows[tid] == rows[tid - j]) {
                val = vals[tid - j];
            }
        }
        __syncthreads();

        
        alpha_add(vals[tid], vals[tid], val);
        __syncthreads();
    }
}


__launch_bounds__(BLOCKSIZE) __global__ static void coomvn_general_block_reduce(
    ALPHA_INT nnz,
    const ALPHA_INT *__restrict__ row_block_red,
    const ALPHA_Number *__restrict__ val_block_red,
    ALPHA_Number *__restrict__ y)
{
    ALPHA_INT tid = hipThreadIdx_x;

    
    if (tid >= nnz) {
        return;
    }

    
    __shared__ ALPHA_INT shared_row[BLOCKSIZE];
    __shared__ ALPHA_Number shared_val[BLOCKSIZE];

    
    for (ALPHA_INT i = tid; i < nnz; i += BLOCKSIZE) {
        
        shared_row[tid] = row_block_red[i];
        shared_val[tid] = val_block_red[i];

        __syncthreads();

        
        segmented_blockreduce(shared_row, shared_val);

        
        ALPHA_INT row   = shared_row[tid];
        ALPHA_INT rowp1 = (tid < BLOCKSIZE - 1) ? shared_row[tid + 1] : -1;

        if (row != rowp1 && row >= 0) {
            
            alpha_add(y[row], y[row], shared_val[tid]);
        }

        __syncthreads();
    }
}

__launch_bounds__(1024) __global__ static void mulbeta(ALPHA_INT m,
                                                       const ALPHA_Number beta,
                                                       ALPHA_Number *__restrict__ y)
{
    ALPHA_INT tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (tid >= m) return;

    alpha_mul(y[tid], y[tid], beta);
}

#ifdef __cplusplus
extern "C" {
#endif

alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      ALPHA_INT m,
      ALPHA_INT n,
      ALPHA_INT nnz,
      const ALPHA_Number alpha,
      const ALPHA_Number *coo_val,
      const ALPHA_INT *coo_row_ind,
      const ALPHA_INT *coo_col_ind,
      const ALPHA_Number *x,
      const ALPHA_Number beta,
      ALPHA_Number *y)
{
    const ALPHA_INT threadPerBlock = 1024;
    const ALPHA_INT blockPerGrid   = m / threadPerBlock + 1;
    hipLaunchKernelGGL(mulbeta, dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, m, beta, y);

#define COOMVN_DIM 128
    ALPHA_INT maxthreads = handle->properties.maxThreadsPerBlock;
    ALPHA_INT nprocs     = handle->properties.multiProcessorCount;
    ALPHA_INT maxblocks  = (nprocs * maxthreads - 1) / COOMVN_DIM + 1;

    const ALPHA_INT wavefront_size = 64;

    ALPHA_INT minblocks = (nnz - 1) / COOMVN_DIM + 1;
    ALPHA_INT nblocks   = maxblocks < minblocks ? maxblocks : minblocks;
    ALPHA_INT nwfs      = nblocks * (COOMVN_DIM / wavefront_size);
    ALPHA_INT nloops    = (nnz / wavefront_size + 1) / nwfs + 1;

    dim3 coomvn_blocks(nblocks);
    dim3 coomvn_threads(COOMVN_DIM);

    
    char *ptr = (char *)(handle->buffer);
    ptr += 256;

    
    ALPHA_INT *row_block_red = (ALPHA_INT *)(ptr);
    ptr += ((sizeof(ALPHA_INT) * nwfs - 1) / 256 + 1) * 256;

    
    ALPHA_Number *val_block_red = (ALPHA_Number *)ptr;

    
    hipLaunchKernelGGL(coomvn_general_wf_reduce,
                       coomvn_blocks,
                       coomvn_threads,
                       0,
                       handle->stream,
                       nnz,
                       nloops,
                       alpha,
                       coo_row_ind,
                       coo_col_ind,
                       coo_val,
                       x,
                       y,
                       row_block_red,
                       val_block_red);

    hipLaunchKernelGGL(coomvn_general_block_reduce,
                       dim3(1),
                       coomvn_threads,
                       0,
                       handle->stream,
                       nwfs,
                       row_block_red,
                       val_block_red,
                       y);
#undef COOMVN_DIM

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif
