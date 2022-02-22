#include "alphasparse/kernel.h"

alphasparse_status_t ONAME(const ALPHA_SPMAT_CSR *A, const ALPHA_SPMAT_CSR *B, ALPHA_SPMAT_CSR **matC)
{
    ALPHA_SPMAT_CSR *transposed_mat;
    transpose_csr(A, &transposed_mat);
    alphasparse_status_t status = spmm_csr(transposed_mat, B, matC);
    destroy_csr(transposed_mat);
    return status;
}