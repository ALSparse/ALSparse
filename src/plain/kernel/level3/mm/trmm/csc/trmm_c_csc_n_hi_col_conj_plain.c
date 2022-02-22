#include "alphasparse/kernel_plain.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_CSC *conjugated_mat;
    transpose_conj_csc(mat, &conjugated_mat);
    alphasparse_status_t status = trmm_csc_n_lo_col_plain(alpha, conjugated_mat, x, columns, ldx, beta, y, ldy);
    destroy_csc(conjugated_mat);
    return status;
}
