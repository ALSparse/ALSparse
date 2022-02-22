#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_CSC *transposed_mat;
    transpose_csc(A, &transposed_mat);
    alphasparse_status_t status = trsm_csc_u_lo_row(alpha, transposed_mat, x, columns, ldx, y, ldy);
    destroy_csc(transposed_mat);
    return status;
}
