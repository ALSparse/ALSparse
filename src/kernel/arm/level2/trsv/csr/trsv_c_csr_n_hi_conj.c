#include "alphasparse/kernel.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSR *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    ALPHA_SPMAT_CSR *conjugated_mat;
    transpose_conj_csr(A, &conjugated_mat);
    alphasparse_status_t status = trsv_csr_n_lo(alpha, conjugated_mat, x, y);
    destroy_csr(conjugated_mat);
    return status;
}
