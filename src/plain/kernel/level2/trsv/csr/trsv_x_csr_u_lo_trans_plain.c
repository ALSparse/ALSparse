#include "alphasparse/kernel_plain.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSR *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    ALPHA_SPMAT_CSR *transposed_mat;
    transpose_csr(A, &transposed_mat);
    alphasparse_status_t status = trsv_csr_u_hi_plain(alpha, transposed_mat, x, y);
    destroy_csr(transposed_mat);
    return status;
}
