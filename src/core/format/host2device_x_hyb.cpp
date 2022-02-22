
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
    if (((alphasparse_matrix*)mtx)->format != ALPHA_SPARSE_FORMAT_HYB) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    ALPHA_SPMAT_HYB* A = (ALPHA_SPMAT_HYB*)((alphasparse_matrix*)mtx)->mat;
    
    if (!A || !A->ell_val || !A->ell_col_ind || !A->coo_val || !A->coo_row_val || !A->coo_col_val)
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;

    ALPHA_INT ell_nnz = A->rows * A->ell_width;

    hipMalloc(&A->d_ell_val, sizeof(ALPHA_Number) * ell_nnz);
    hipMalloc(&A->d_ell_col_ind, sizeof(ALPHA_INT) * ell_nnz);

    hipMalloc(&A->d_coo_val, sizeof(ALPHA_Number) * A->nnz);
    hipMalloc(&A->d_coo_row_val, sizeof(ALPHA_INT) * A->nnz);
    hipMalloc(&A->d_coo_col_val, sizeof(ALPHA_INT) * A->nnz);


    hipMemcpy(A->d_ell_val, A->ell_val, sizeof(ALPHA_Number) * ell_nnz, hipMemcpyHostToDevice);
    hipMemcpy(A->d_ell_col_ind, A->ell_col_ind, sizeof(ALPHA_INT) * ell_nnz, hipMemcpyHostToDevice);

    hipMemcpy(A->d_coo_val, A->coo_val, sizeof(ALPHA_Number) * A->nnz, hipMemcpyHostToDevice);
    hipMemcpy(A->d_coo_row_val, A->coo_row_val, sizeof(ALPHA_INT) * A->nnz, hipMemcpyHostToDevice);
    hipMemcpy(A->d_coo_col_val, A->coo_col_val, sizeof(ALPHA_INT) * A->nnz, hipMemcpyHostToDevice);
#else
    return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
#endif
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
#ifdef __cplusplus
}
#endif