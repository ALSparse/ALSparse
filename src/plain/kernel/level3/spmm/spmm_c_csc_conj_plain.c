#include "alphasparse/kernel_plain.h"
#include <stdio.h>


alphasparse_status_t ONAME(const ALPHA_SPMAT_CSC *A, const ALPHA_SPMAT_CSC *B, ALPHA_SPMAT_CSC **matC)
{
    ALPHA_SPMAT_CSC *conjugated_mat;
    transpose_conj_csc(A, &conjugated_mat);
    alphasparse_status_t status = spmm_csc_plain(conjugated_mat, B, matC);
    destroy_csc(conjugated_mat);
    return status;
}
