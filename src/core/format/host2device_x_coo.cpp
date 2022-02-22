
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
    if (((alphasparse_matrix*)mtx)->format != ALPHA_SPARSE_FORMAT_COO) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    ALPHA_SPMAT_COO* A = (ALPHA_SPMAT_COO*)((alphasparse_matrix*)mtx)->mat;
    
    if (!A || !A->row_indx || !A->col_indx || !A->values)
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;

    if (!A->ordered)
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;

    ALPHA_INT nnz = A->nnz;

    hipMalloc(&A->d_rows_indx, sizeof(ALPHA_INT) * nnz);
    hipMalloc(&A->d_cols_indx, sizeof(ALPHA_INT) * nnz);
    hipMalloc(&A->d_values, sizeof(ALPHA_Number) * nnz);

    hipMemcpy(A->d_rows_indx, A->row_indx, sizeof(ALPHA_INT) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(A->d_cols_indx, A->col_indx, sizeof(ALPHA_INT) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(A->d_values, A->values, sizeof(ALPHA_Number) * nnz, hipMemcpyHostToDevice);
#else
    return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
#endif
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
#ifdef __cplusplus
}
#endif