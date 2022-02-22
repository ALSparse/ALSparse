#include "alphasparse/kernel_plain.h"

alphasparse_status_t ONAME(const ALPHA_SPMAT_CSC *matA, const ALPHA_SPMAT_CSC *matB, ALPHA_Number *matC, const ALPHA_INT ldc)
{
    ALPHA_SPMAT_CSC *transposed_mat;
    transpose_csc(matA, &transposed_mat);
    alphasparse_status_t status = spmmd_csc_col_plain(transposed_mat,matB,matC,ldc);
    destroy_csc(transposed_mat);
    return status;
}
