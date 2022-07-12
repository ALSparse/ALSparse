
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
    if (((alphasparse_matrix*)mtx)->format != ALPHA_SPARSE_FORMAT_CSR) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    ALPHA_SPMAT_CSR* A = (ALPHA_SPMAT_CSR*)((alphasparse_matrix*)mtx)->mat;
    
    if (!A || !A->rows_start || !A->rows_end || !A->col_indx || !A->values)
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;

    ALPHA_INT nnz = A->rows_end[A->rows - 1];

    hipMalloc(&A->d_row_ptr, sizeof(ALPHA_INT) * (A->rows + 1));
    hipMalloc(&A->d_col_indx, sizeof(ALPHA_INT) * nnz);
    hipMalloc(&A->d_values, sizeof(ALPHA_Number) * nnz);

    hipMemcpy(A->d_row_ptr, A->rows_start, sizeof(ALPHA_INT) * (A->rows + 1), hipMemcpyHostToDevice);
    hipMemcpy(A->d_col_indx, A->col_indx, sizeof(ALPHA_INT) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(A->d_values, A->values, sizeof(ALPHA_Number) * nnz, hipMemcpyHostToDevice);
#else
    return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
#endif
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
#ifdef __cplusplus
}
#endif