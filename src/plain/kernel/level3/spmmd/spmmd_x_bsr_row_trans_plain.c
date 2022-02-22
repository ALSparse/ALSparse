#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

alphasparse_status_t ONAME(const ALPHA_SPMAT_BSR *matA, const ALPHA_SPMAT_BSR *matB, ALPHA_Number *matC, const ALPHA_INT ldc)
{
    ALPHA_SPMAT_BSR *transposed_mat;
    transpose_bsr(matA, &transposed_mat);
    alphasparse_status_t status = spmmd_bsr_row_plain(transposed_mat,matB,matC,ldc);
    destroy_bsr(transposed_mat);
    return status;
}
