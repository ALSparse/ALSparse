#include "alphasparse/kernel.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    ALPHA_SPMAT_COO *conjugated_mat;
    transpose_conj_coo(A, &conjugated_mat);
    alphasparse_status_t status = trsv_coo_u_lo(alpha, conjugated_mat, x, y);
    destroy_coo(conjugated_mat);
    return status;
}
