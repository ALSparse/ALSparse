#include "alphasparse/kernel.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_CSC *transposed_mat;
    transpose_csc(mat, &transposed_mat);
    alphasparse_status_t status = trmm_csc_n_lo_col(alpha, transposed_mat, x, columns, ldx, beta, y, ldy);
    destroy_csc(transposed_mat);
    return status;
}
