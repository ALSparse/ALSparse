#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_DIA *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_DIA *conjugated_mat;
    transpose_conj_dia(A, &conjugated_mat);
    alphasparse_status_t status = trsm_dia_u_hi_col_plain(alpha, conjugated_mat, x, columns, ldx, y, ldy);
    destroy_dia(conjugated_mat);
    return status;
}
