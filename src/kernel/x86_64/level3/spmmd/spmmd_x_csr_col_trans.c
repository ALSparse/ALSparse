#include "alphasparse/kernel.h"

alphasparse_status_t ONAME(const ALPHA_SPMAT_CSR *matA, const ALPHA_SPMAT_CSR *matB, ALPHA_Number *matC, const ALPHA_INT ldc)
{
    ALPHA_SPMAT_CSR *transposed_mat;
    transpose_csr(matA, &transposed_mat);
    alphasparse_status_t status = spmmd_csr_col(transposed_mat, matB , matC, ldc);
    destroy_csr(transposed_mat);
    return status;
}
