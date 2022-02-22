#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_BSR *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_BSR *conjugated_mat;
    transpose_conj_bsr(A, &conjugated_mat);
    alphasparse_status_t status = trsm_bsr_n_lo_col_plain(alpha, conjugated_mat, x, columns, ldx, y, ldy);
    destroy_bsr(conjugated_mat);
    return status;
}
