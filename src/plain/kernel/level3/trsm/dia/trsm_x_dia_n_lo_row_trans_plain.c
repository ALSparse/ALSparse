#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_DIA *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_DIA *transposed_mat;
    transpose_dia(A, &transposed_mat);
    alphasparse_status_t status = trsm_dia_n_hi_row_plain(alpha, transposed_mat, x, columns, ldx, y, ldy);
    destroy_dia(transposed_mat);
    return status;
}
