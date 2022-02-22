#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_BSR *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_BSR *transposed_mat;
    transpose_bsr(mat, &transposed_mat);
    alphasparse_status_t status = gemm_bsr_col(alpha, transposed_mat, x, columns, ldx, beta, y, ldy);
    destroy_bsr(transposed_mat);
    return status;
}
