#include "alphasparse/kernel_plain.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    ALPHA_SPMAT_CSC *transposed_mat;
    transpose_csc(A, &transposed_mat);
    alphasparse_status_t status = trsv_csc_n_lo_plain(alpha, transposed_mat, x, y);
    destroy_csc(transposed_mat);
    return status;
}
