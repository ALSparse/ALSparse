#include "alphasparse/kernel.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_COO *conjugated_mat;
    transpose_conj_coo(A, &conjugated_mat);
    alphasparse_status_t status = trsm_coo_u_hi_row(alpha, conjugated_mat, x, columns, ldx, y, ldy);
    destroy_coo(conjugated_mat);
    return status;
}
