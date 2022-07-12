
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
    if (((alphasparse_matrix*)mtx)->format != ALPHA_SPARSE_FORMAT_ELL) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    ALPHA_SPMAT_ELL* A = (ALPHA_SPMAT_ELL*)((alphasparse_matrix*)mtx)->mat;
    
    if (!A || !A->values || !A->indices)
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;

    ALPHA_INT m  = A->rows;
    ALPHA_INT ld = A->ld;

    hipMalloc(&A->d_indices, sizeof(ALPHA_INT) * ld * m);
    hipMalloc(&A->d_values, sizeof(ALPHA_Number) * ld * m);

    hipMemcpy(A->d_indices, A->indices, sizeof(ALPHA_INT) * ld * m, hipMemcpyHostToDevice);
    hipMemcpy(A->d_values, A->values, sizeof(ALPHA_Number) * ld * m, hipMemcpyHostToDevice);
#else
    return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
#endif
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
#ifdef __cplusplus
}
#endif