#include "alphasparse/kernel.h"
#include "alphasparse/util.h"


alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSR *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_CSR *transposed_mat;
    transpose_csr(mat, &transposed_mat);
    alphasparse_status_t status = gemm_csr_row(alpha, transposed_mat, x, columns, ldx, beta, y, ldy);
    destroy_csr(transposed_mat);
    return status;
}
