#include "alphasparse/kernel.h"

alphasparse_status_t ONAME(const ALPHA_SPMAT_CSR *matA, const ALPHA_SPMAT_CSR *matB, ALPHA_Number *matC, const ALPHA_INT ldc)
{
    ALPHA_SPMAT_CSR *conjugated_mat;
    transpose_conj_csr(matA, &conjugated_mat);
    alphasparse_status_t status = spmmd_csr_row(conjugated_mat, matB , matC, ldc);
    destroy_csr(conjugated_mat);
    return status;
}
