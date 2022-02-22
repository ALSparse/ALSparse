#include "alphasparse/kernel.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_BSR *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    ALPHA_SPMAT_BSR *conjugated_mat;
    transpose_conj_bsr(A, &conjugated_mat);
    alphasparse_status_t status = trsv_bsr_u_lo(alpha, conjugated_mat, x, y);
    destroy_bsr(conjugated_mat);
    return status;
}
