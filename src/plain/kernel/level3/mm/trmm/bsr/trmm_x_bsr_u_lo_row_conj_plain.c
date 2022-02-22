#include "alphasparse/kernel_plain.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_BSR *mat, const ALPHA_Number *x, const ALPHA_INT rowumns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
#ifdef COMPLEX
    ALPHA_SPMAT_BSR *conjugated_mat;
    transpose_conj_bsr(mat, &conjugated_mat);
    alphasparse_status_t status = trmm_bsr_u_hi_row_plain(alpha, conjugated_mat, x, rowumns, ldx, beta, y, ldy);
    destroy_bsr(conjugated_mat);
    return status;
#else
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
#endif

}
