#include "alphasparse/kernel.h"

alphasparse_status_t ONAME(const ALPHA_SPMAT_BSR *A, const ALPHA_SPMAT_BSR *B, ALPHA_SPMAT_BSR **matC)
{
    ALPHA_SPMAT_BSR *conjugated_mat;
    transpose_conj_bsr(A, &conjugated_mat);
    alphasparse_status_t status = spmm_bsr(conjugated_mat, B, matC);
    destroy_bsr(conjugated_mat);
    return status;
}
