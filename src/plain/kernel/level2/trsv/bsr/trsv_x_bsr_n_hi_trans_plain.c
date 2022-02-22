#include "alphasparse/kernel_plain.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_BSR *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    ALPHA_SPMAT_BSR *transposed_mat;
    transpose_bsr(A, &transposed_mat);
    alphasparse_status_t status = trsv_bsr_n_lo_plain(alpha, transposed_mat, x, y);
    destroy_bsr(transposed_mat);
    return status;
}
