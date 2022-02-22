#include "alphasparse/kernel.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSR *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_CSR *conjugated_mat;
    transpose_conj_csr(A, &conjugated_mat);
    alphasparse_status_t status = trsm_csr_u_lo_row(alpha, conjugated_mat, x, columns, ldx, y, ldy);
    destroy_csr(conjugated_mat);
    return status;
}
