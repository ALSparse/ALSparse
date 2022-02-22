#include "alphasparse/kernel.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_DIA *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_DIA *transposed_mat;
    transpose_dia(mat, &transposed_mat);
    alphasparse_status_t status = trmm_dia_u_lo_row(alpha, transposed_mat, x, columns, ldx, beta, y, ldy);
    destroy_dia(transposed_mat);
    return status;
}
