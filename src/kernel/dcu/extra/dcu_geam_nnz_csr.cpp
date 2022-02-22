#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/util_dcu.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

#define WFSIZE      64
#define CSRGEAM_DIM 256
#define BLOCKSIZE   CSRGEAM_DIM

__global__ static void
geam_nnz_rocsparse(ALPHA_INT m,
                   ALPHA_INT n,
                   const ALPHA_INT *csr_row_ptr_A,
                   const ALPHA_INT *csr_col_ind_A,
                   const ALPHA_INT *csr_row_ptr_B,
                   const ALPHA_INT *csr_col_ind_B,
                   ALPHA_INT *row_nnz)
{
    // todo: can't find bugs
    // Lane id
    ALPHA_INT lid = hipThreadIdx_x & (WFSIZE - 1);

    // Wavefront id
    ALPHA_INT wid = hipThreadIdx_x / WFSIZE;

    // Each wavefront processes a row
    ALPHA_INT row = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

    // Do not run out of bounds
    if (row >= m) {
        return;
    }

    // Row nnz marker
    __shared__ bool stable[BLOCKSIZE];
    bool *table = &stable[wid * WFSIZE];

    // Get row entry and exit point of A
    ALPHA_INT row_begin_A = csr_row_ptr_A[row];
    ALPHA_INT row_end_A   = csr_row_ptr_A[row + 1];

    // Get row entry and exit point of B
    ALPHA_INT row_begin_B = csr_row_ptr_B[row];
    ALPHA_INT row_end_B   = csr_row_ptr_B[row + 1];

    // Load the first column of the current row from A and B to set the starting
    // point for the first chunk
    ALPHA_INT col_A = (row_begin_A < row_end_A) ? csr_col_ind_A[row_begin_A] : n;
    ALPHA_INT col_B = (row_begin_B < row_end_B) ? csr_col_ind_B[row_begin_B] : n;

    // Begin of the current row chunk
    ALPHA_INT chunk_begin = min(col_A, col_B);

    // Initialize the row nnz for the full (wavefront-wide) row
    ALPHA_INT nnz = 0;

    // Initialize the index for column access into A and B
    row_begin_A += lid;
    row_begin_B += lid;

    // Loop over the chunks until the end of both rows (A and B) has been reached (which
    // is the number of total columns n)
    while (true) {
        // Initialize row nnz table
        table[lid] = false;

        __threadfence_block();

        // Initialize the beginning of the next chunk
        ALPHA_INT min_col = n;

        // Loop over all columns of A, starting with the first entry that did not fit
        // into the previous chunk
        for (; row_begin_A < row_end_A; row_begin_A += WFSIZE) {
            // Get the column of A
            ALPHA_INT col_A = csr_col_ind_A[row_begin_A];

            // Get the column of A shifted by the chunk_begin
            ALPHA_INT shf_A = col_A - chunk_begin;

            // Check if this column of A is within the chunk
            if (shf_A < WFSIZE) {
                // Mark this column in shared memory
                table[shf_A] = true;
            } else {
                // Store the first column index of A that exceeds the current chunk
                min_col = min(min_col, col_A);
                break;
            }
        }

        // Loop over all columns of B, starting with the first entry that did not fit
        // into the previous chunk
        for (; row_begin_B < row_end_B; row_begin_B += WFSIZE) {
            // Get the column of B
            ALPHA_INT col_B = csr_col_ind_B[row_begin_B];

            // Get the column of B shifted by the chunk_begin
            ALPHA_INT shf_B = col_B - chunk_begin;

            // Check if this column of B is within the chunk
            if (shf_B < WFSIZE) {
                // Mark this column in shared memory
                table[shf_B] = true;
            } else {
                // Store the first column index of B that exceeds the current chunk
                min_col = min(min_col, col_B);
                break;
            }
        }

        __threadfence_block();

        // Compute the chunk's number of non-zeros of the row and add it to the global
        // row nnz counter
        nnz += __popcll(__ballot(table[lid]));

        // Gather wavefront-wide minimum for the next chunks starting column index
        // Using shfl_xor here so that each thread in the wavefront obtains the final
        // result
        for (unsigned int i = WFSIZE >> 1; i > 0; i >>= 1) {
            min_col = min(min_col, __shfl_xor(min_col, i));
        }

        // Each thread sets the new chunk beginning
        chunk_begin = min_col;

        // Once the chunk beginning has reached the total number of columns n,
        // we are done
        if (chunk_begin >= n) {
            break;
        }
    }

    // Last thread in each wavefront writes the accumulated total row nnz to global
    // memory
    if (lid == WFSIZE - 1) {
        row_nnz[row] = nnz;
    }
}

__global__ static void
geam_nnz_per_row(ALPHA_INT m,
                 ALPHA_INT n,
                 const ALPHA_INT *csr_row_ptr_A,
                 const ALPHA_INT *csr_col_ind_A,
                 const ALPHA_INT *csr_row_ptr_B,
                 const ALPHA_INT *csr_col_ind_B,
                 ALPHA_INT *row_nnz)
{
    ALPHA_INT tid    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ALPHA_INT stride = hipBlockDim_x * hipGridDim_x;
    row_nnz[0]       = 0;
    for (ALPHA_INT r = tid; r < n; r += stride) {
        row_nnz[r + 1] = 0;

        ALPHA_INT as = csr_row_ptr_A[r];
        ALPHA_INT ae = csr_row_ptr_A[r + 1];
        ALPHA_INT bs = csr_row_ptr_B[r];
        ALPHA_INT be = csr_row_ptr_B[r + 1];

        ALPHA_INT ai = as, bi = bs;
        while (ai < ae && bi < be) {
            ALPHA_INT ac = csr_col_ind_A[ai];
            ALPHA_INT bc = csr_col_ind_B[bi];
            if (ac < bc) {
                ai++;
            } else if (ac > bc) {
                bi++;
            } else {
                ai++;
                bi++;
            }
            row_nnz[r + 1]++;
        }
        if (ai == ae) {
            row_nnz[r + 1] += be - bi;
        } else {
            row_nnz[r + 1] += ae - ai;
        }
    }
}

__global__ static void
prefix(ALPHA_INT *row_nnz, ALPHA_INT size)
{
    // todo: opt
    ALPHA_INT tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (tid == 0)
        for (int i = 1; i < size; i++) {
            row_nnz[i] += row_nnz[i - 1];
        }
}

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

alphasparse_status_t
dcu_geam_nnz_csr(alphasparse_dcu_handle_t handle,
                 ALPHA_INT m,
                 ALPHA_INT n,
                 ALPHA_INT nnz_A,
                 const ALPHA_INT *csr_row_ptr_A,
                 const ALPHA_INT *csr_col_ind_A,
                 ALPHA_INT nnz_B,
                 const ALPHA_INT *csr_row_ptr_B,
                 const ALPHA_INT *csr_col_ind_B,
                 ALPHA_INT *csr_row_ptr_C,
                 ALPHA_INT *nnz_C)
{
    const ALPHA_INT threadPerBlock = 256;
    const int blockPerGrid         = min(32, (threadPerBlock + n - 1) / threadPerBlock);

    // printf("geam nnz \n");
    hipLaunchKernelGGL(geam_nnz_per_row, blockPerGrid, threadPerBlock, 0, handle->stream, m, n, csr_row_ptr_A, csr_col_ind_A, csr_row_ptr_B, csr_col_ind_B, csr_row_ptr_C);

    hipDeviceSynchronize();

    hipLaunchKernelGGL(prefix, dim3(1), dim3(1), 0, handle->stream, csr_row_ptr_C, m + 1);

    hipMemcpyAsync(nnz_C, csr_row_ptr_C + m, sizeof(ALPHA_INT), hipMemcpyDeviceToDevice);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
