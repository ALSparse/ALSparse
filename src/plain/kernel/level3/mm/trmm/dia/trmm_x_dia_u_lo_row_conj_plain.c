#include "alphasparse/kernel_plain.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_DIA *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
#ifdef COMPLEX
    ALPHA_SPMAT_DIA *conjugated_mat;
    transpose_conj_dia(mat, &conjugated_mat);
    alphasparse_status_t status = trmm_dia_u_hi_row_plain(alpha, conjugated_mat, x, columns, ldx, beta, y, ldy);
    destroy_dia(conjugated_mat);
    return status;
#else
	return ALPHA_SPARSE_STATUS_INVALID_VALUE;
#endif
}
