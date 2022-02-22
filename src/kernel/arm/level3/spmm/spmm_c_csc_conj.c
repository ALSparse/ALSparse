#include "alphasparse/kernel.h"

alphasparse_status_t ONAME(const ALPHA_SPMAT_CSC *A, const ALPHA_SPMAT_CSC *B, ALPHA_SPMAT_CSC **matC)
{
    ALPHA_SPMAT_CSC *conjugated_mat;
    transpose_conj_csc(A, &conjugated_mat);
    alphasparse_status_t status = spmm_csc(conjugated_mat, B, matC);
    destroy_csc(conjugated_mat);
    return status;
}
