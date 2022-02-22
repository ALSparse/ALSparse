#include "alphasparse/kernel.h"

alphasparse_status_t ONAME(const ALPHA_SPMAT_BSR *A, const ALPHA_SPMAT_BSR *B, ALPHA_SPMAT_BSR **matC)
{
    ALPHA_SPMAT_BSR *transposed_mat;
    transpose_bsr(A, &transposed_mat);
    alphasparse_status_t status = spmm_bsr(transposed_mat, B, matC);
    destroy_bsr(transposed_mat);
    return status;
}
