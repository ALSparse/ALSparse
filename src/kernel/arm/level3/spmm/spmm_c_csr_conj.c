#include "alphasparse/kernel.h"

alphasparse_status_t ONAME(const ALPHA_SPMAT_CSR *A, const ALPHA_SPMAT_CSR *B, ALPHA_SPMAT_CSR **matC)
{
    ALPHA_SPMAT_CSR *conjugated_mat;
    transpose_conj_csr(A, &conjugated_mat);
    alphasparse_status_t status = spmm_csr(conjugated_mat, B, matC);
    destroy_csr(conjugated_mat);
    return status;
}