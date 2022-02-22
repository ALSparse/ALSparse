#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_DIA *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_DIA *conjugated_mat;
    transpose_conj_dia(mat, &conjugated_mat);
    alphasparse_status_t status = gemm_dia_col(alpha, conjugated_mat, x, columns, ldx, beta, y, ldy);
    destroy_dia(conjugated_mat);
    return status;
}
