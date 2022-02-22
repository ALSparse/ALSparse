#include "alphasparse/kernel_plain.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_SKY *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    // ALPHA_SPMAT_SKY *transposed_mat;
    // transpose_sky(mat, &transposed_mat);
    // alphasparse_status_t status = trmm_sky_n_lo_row_plain(alpha, transposed_mat, x, columns, ldx, beta, y, ldy);
    // destroy_sky(transposed_mat);
    alphasparse_status_t status = trmm_sky_n_lo_row_plain(alpha, mat, x, columns, ldx, beta, y, ldy);
    return status;
}
