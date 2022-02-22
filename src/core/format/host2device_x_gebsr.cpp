
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
    if (((alphasparse_matrix*)mtx)->format != ALPHA_SPARSE_FORMAT_GEBSR) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    ALPHA_SPMAT_GEBSR* A = (ALPHA_SPMAT_GEBSR*)((alphasparse_matrix*)mtx)->mat;
    
    if (!A || !A->values || !A->rows_start || !A->rows_end || !A->col_indx)
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;

    if (!A->ordered)
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;

    ALPHA_INT block_nnz = A->rows_end[A->rows - 1];
    ALPHA_INT nnz       = block_nnz * A->row_block_dim * A->col_block_dim;

    hipMalloc(&A->d_rows_ptr, sizeof(ALPHA_INT) * (A->rows + 1));
    hipMalloc(&A->d_col_indx, sizeof(ALPHA_INT) * block_nnz);
    hipMalloc(&A->d_values, sizeof(ALPHA_Number) * nnz);

    hipMemcpy(A->d_rows_ptr, A->rows_start, sizeof(ALPHA_INT) * (A->rows + 1), hipMemcpyHostToDevice);
    hipMemcpy(A->d_col_indx, A->col_indx, sizeof(ALPHA_INT) * block_nnz, hipMemcpyHostToDevice);
    hipMemcpy(A->d_values, A->values, sizeof(ALPHA_Number) * nnz, hipMemcpyHostToDevice);
#else
    return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
#endif
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
#ifdef __cplusplus
}
#endif