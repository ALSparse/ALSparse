#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_CSC *conjugated_mat;
    transpose_conj_csc(A, &conjugated_mat);
    alphasparse_status_t status = trsm_csc_n_hi_row(alpha, conjugated_mat, x, columns, ldx, y, ldy);
    destroy_csc(conjugated_mat);
    return status;
}
