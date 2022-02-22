#include "alphasparse/kernel_plain.h"

alphasparse_status_t ONAME(const ALPHA_SPMAT_CSC *matA, const ALPHA_SPMAT_CSC *matB, ALPHA_Number *matC, const ALPHA_INT ldc)
{
    ALPHA_SPMAT_CSC *conjugated_mat;
    transpose_conj_csc(matA, &conjugated_mat);
    alphasparse_status_t status = spmmd_csc_col_plain(conjugated_mat,matB,matC,ldc);
    destroy_csc(conjugated_mat);
    return status;
}
