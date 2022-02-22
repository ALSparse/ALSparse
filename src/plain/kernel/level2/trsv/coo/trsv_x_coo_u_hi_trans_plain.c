#include "alphasparse/kernel_plain.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    ALPHA_SPMAT_COO *transposed_mat;
    transpose_coo(A, &transposed_mat);
    alphasparse_status_t status = trsv_coo_u_lo_plain(alpha, transposed_mat, x, y);
    destroy_coo(transposed_mat);
    return status;
}
