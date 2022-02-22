
#include "alphasparse/util.h"
#include "alphasparse/format.h"
#ifdef __DCU__
#include <hip/hip_runtime_api.h>
#endif

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

alphasparse_status_t ONAME(alphasparse_matrix_t mtx)

{
#ifdef __DCU__
    if (!mtx) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }
    if (((alphasparse_matrix*)mtx)->format != ALPHA_SPARSE_FORMAT_CSR5) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    ALPHA_SPMAT_CSR5* A = (ALPHA_SPMAT_CSR5*)((alphasparse_matrix*)mtx)->mat;
    // printf("check csr5 concent\n");
    // printf("%d %d %d\n", A->num_rows, A->num_cols, A->nnz);
    // printf("%d %d %d %d %d %d %d\n", A->csr5_sigma, A->csr5_bit_y_offset, A->csr5_bit_scansum_offset, A->csr5_num_packets, A->csr5_p, A->csr5_num_offsets, A->csr5_tail_tile_start);     // opt: info for CSR5);
    
    if (!A || !A->row_ptr || !A->col_idx || !A->val || !A->tile_ptr || !A->tile_desc || !A->tile_desc_offset_ptr || !A->calibrator)
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;

    hipMalloc(&A->d_row_ptr, sizeof(ALPHA_INT) * (A->num_rows + 1));
    hipMalloc(&A->d_col_idx, sizeof(ALPHA_INT) * A->nnz);
    hipMalloc(&A->d_val, sizeof(ALPHA_Number) * A->nnz);

    hipMalloc(&A->d_tile_ptr, sizeof(uint32_t) * (A->csr5_p + 1));
    hipMalloc(&A->d_tile_desc, sizeof(uint32_t) * (A->csr5_p * ALPHA_CSR5_OMEGA * A->csr5_num_packets));
    hipMalloc(&A->d_tile_desc_offset_ptr, sizeof(ALPHA_INT) * (A->csr5_p+1));
    hipMalloc(&A->d_tile_desc_offset, sizeof(ALPHA_INT) * A->csr5_num_offsets);
    hipMalloc(&A->d_calibrator, sizeof(ALPHA_Number) * A->csr5_p);

    hipMemcpy(A->d_row_ptr, A->row_ptr, sizeof(ALPHA_INT) * (A->num_rows + 1), hipMemcpyHostToDevice);
    hipMemcpy(A->d_col_idx, A->col_idx, sizeof(ALPHA_INT) * A->nnz, hipMemcpyHostToDevice);
    hipMemcpy(A->d_val, A->val, sizeof(ALPHA_Number) * A->nnz, hipMemcpyHostToDevice);

    hipMemcpy(A->d_tile_ptr, A->tile_ptr, sizeof(uint32_t) * (A->csr5_p + 1), hipMemcpyHostToDevice);
    hipMemcpy(A->d_tile_desc, A->tile_desc, sizeof(uint32_t) * (A->csr5_p * ALPHA_CSR5_OMEGA * A->csr5_num_packets), hipMemcpyHostToDevice);
    hipMemcpy(A->d_tile_desc_offset_ptr, A->tile_desc_offset_ptr, sizeof(ALPHA_INT) * (A->csr5_p+1), hipMemcpyHostToDevice);
    hipMemcpy(A->d_tile_desc_offset, A->tile_desc_offset, sizeof(ALPHA_INT) * A->csr5_num_offsets, hipMemcpyHostToDevice);
    hipMemcpy(A->d_calibrator, A->calibrator, sizeof(ALPHA_Number) * A->csr5_p, hipMemcpyHostToDevice);

#else
    return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
#endif
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
#ifdef __cplusplus
}
#endif