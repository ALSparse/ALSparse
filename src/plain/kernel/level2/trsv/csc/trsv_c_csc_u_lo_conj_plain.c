#include "alphasparse/kernel_plain.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    ALPHA_SPMAT_CSC *conjugated_mat;
    transpose_conj_csc(A, &conjugated_mat);
    alphasparse_status_t status = trsv_csc_u_hi_plain(alpha, conjugated_mat, x, y);
    destroy_csc(conjugated_mat);
    return status;
}
