#include "alphasparse/handle.h"
#include <hip/hip_runtime.h>
#include "alphasparse/common_dcu.h"
#include "alphasparse/util_dcu.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

__global__ static void
spgemm_blk(ALPHA_INT m,
           ALPHA_INT n,
           ALPHA_INT k,
           const ALPHA_Number alpha,
           const ALPHA_Number *csr_val_A,
           const ALPHA_INT *csr_row_ptr_A,
           const ALPHA_INT *csr_col_ind_A,
           const ALPHA_Number *csr_val_B,
           const ALPHA_INT *csr_row_ptr_B,
           const ALPHA_INT *csr_col_ind_B,
           const ALPHA_Number beta,
           const ALPHA_Number *csr_val_D,
           const ALPHA_INT *csr_row_ptr_D,
           const ALPHA_INT *csr_col_ind_D,
           ALPHA_Number *csr_val_C,
           const ALPHA_INT *csr_row_ptr_C,
           ALPHA_INT *csr_col_ind_C)
{
    ALPHA_INT tid    = hipThreadIdx_x;
    ALPHA_INT stride = hipBlockDim_x;

    extern __shared__ char shr[];
    ALPHA_Number *values  = reinterpret_cast<ALPHA_Number *>(shr);
    ALPHA_INT *write_back = reinterpret_cast<ALPHA_INT *>(values + n);

    //for (ALPHA_INT ar = tid; ar < m; ar += stride)
    ALPHA_INT ar = hipBlockIdx_x;
    {
        for (ALPHA_INT i = tid; i < n; i += stride) {
            alpha_setzero(values[i]);
            write_back[i] = 0;
        }
        __syncthreads();

        for (ALPHA_INT di = csr_row_ptr_D[ar] + tid; di < csr_row_ptr_D[ar + 1]; di += stride) {
            ALPHA_INT dc = csr_col_ind_D[di];
            alpha_mul(values[dc], beta, csr_val_D[di]);
            write_back[dc] = 1;
        }
        __syncthreads();

        for (ALPHA_INT ai = csr_row_ptr_A[ar] + tid; ai < csr_row_ptr_A[ar + 1]; ai += stride) {
            ALPHA_INT br    = csr_col_ind_A[ai];
            ALPHA_Number av = csr_val_A[ai];
            ALPHA_Number tmp;
            alpha_mul(tmp, alpha, csr_val_A[ai]);

            for (ALPHA_INT bi = csr_row_ptr_B[br]; bi < csr_row_ptr_B[br + 1]; bi++) {
                ALPHA_INT bc    = csr_col_ind_B[bi];
                ALPHA_Number bv = csr_val_B[bi];

                //alpha_madde(values[bc], tmp, bv);
                ALPHA_Number t;
                alpha_mul(t, tmp, bv);
                alpha_atomic_add(values[bc], t);

                write_back[bc] = 1;
            }
        }
        __syncthreads();

        // in-place prefix sum
        ALPHA_INT n64    = 1;
        ALPHA_INT stop   = 2;
        ALPHA_INT t_stop = 1;
        ALPHA_INT i;

        while (n64 < n)
            n64 = n64 << 1;
        n64 = n64 >> 1;

        if (n64 != 0) {
            while (stop <= n64) {
                for (i = tid; i < n64; i += stride) {
                    if (i % stop >= t_stop)
                        write_back[i] += write_back[i - i % t_stop - 1];
                }
                __syncthreads();

                stop   = stop << 1;
                t_stop = t_stop << 1;
            }
        } else
            n64++;

        if (tid == 0) {
            for (ALPHA_INT i = n64; i < n; i++) {
                write_back[i] = write_back[i] + write_back[i - 1];
            }
        }

        __syncthreads();

        ALPHA_INT index = csr_row_ptr_C[ar];
        for (ALPHA_INT c = tid; c < n; c += stride) {
            if (c == 0 && write_back[c]) {
                csr_col_ind_C[index] = c;
                csr_val_C[index]     = values[c];
            } else {
                if (write_back[c] - write_back[c - 1]) {
                    csr_col_ind_C[index + write_back[c] - 1] = c;
                    csr_val_C[index + write_back[c] - 1]     = values[c];
                }
            }
        }

        //TODO scan + write
        // if (tid == 0)
        // {
        //     ALPHA_INT index = csr_row_ptr_C[ar];
        //     for (ALPHA_INT c = 0; c < n; c++)
        //     {
        //         if (write_back[c])
        //         {
        //             csr_col_ind_C[index] = c;
        //             csr_val_C[index] = values[c];
        //             index += 1;
        //         }
        //     }
        // }
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
      const ALPHA_Number alpha,
      ALPHA_INT nnz_A,
      const ALPHA_Number *csr_val_A,
      const ALPHA_INT *csr_row_ptr_A,
      const ALPHA_INT *csr_col_ind_A,
      ALPHA_INT nnz_B,
      const ALPHA_Number *csr_val_B,
      const ALPHA_INT *csr_row_ptr_B,
      const ALPHA_INT *csr_col_ind_B,
      const ALPHA_Number beta,
      ALPHA_INT nnz_D,
      const ALPHA_Number *csr_val_D,
      const ALPHA_INT *csr_row_ptr_D,
      const ALPHA_INT *csr_col_ind_D,
      ALPHA_Number *csr_val_C,
      const ALPHA_INT *csr_row_ptr_C,
      ALPHA_INT *csr_col_ind_C,
      const alphasparse_dcu_mat_info_t info_C,
      void *temp_buffer)
{
    const ALPHA_INT threadPerBlock = 256;
    const ALPHA_INT blockPerGrid   = (m - 1) / threadPerBlock + 1;

    hipLaunchKernelGGL(spgemm_blk, m, 256, n * (sizeof(ALPHA_Number) + sizeof(ALPHA_INT)), handle->stream, m, n, k, alpha, csr_val_A, csr_row_ptr_A, csr_col_ind_A, csr_val_B, csr_row_ptr_B, csr_col_ind_B, beta, csr_val_D, csr_row_ptr_D, csr_col_ind_D, csr_val_C, csr_row_ptr_C, csr_col_ind_C);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
