#include "alphasparse/kernel_plain.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_COO *conjugated_mat;
    transpose_conj_coo(mat, &conjugated_mat);
    alphasparse_status_t status = trmm_coo_n_hi_row_plain(alpha, conjugated_mat, x, columns, ldx, beta, y, ldy);
    destroy_coo(conjugated_mat);
    return status;
}
