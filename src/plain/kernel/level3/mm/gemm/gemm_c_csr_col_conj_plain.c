#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"


alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSR *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_CSR *conjugated_mat;
    transpose_conj_csr(mat, &conjugated_mat);
    alphasparse_status_t status = gemm_csr_col_plain(alpha, conjugated_mat, x, columns, ldx, beta, y, ldy);
    destroy_csr(conjugated_mat);
    return status;
}
