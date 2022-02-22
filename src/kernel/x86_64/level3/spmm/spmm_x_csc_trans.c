#include "alphasparse/kernel.h"

alphasparse_status_t ONAME(const ALPHA_SPMAT_CSC *A, const ALPHA_SPMAT_CSC *B, ALPHA_SPMAT_CSC **matC)
{
    ALPHA_SPMAT_CSC *transposed_mat;
    transpose_csc(A, &transposed_mat);
    alphasparse_status_t status = spmm_csc(transposed_mat, B, matC);
    destroy_csc(transposed_mat);
    return status;
}
