#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

alphasparse_status_t ONAME(const ALPHA_SPMAT_BSR *matA, const ALPHA_SPMAT_BSR *matB, ALPHA_Number *matC, const ALPHA_INT ldc)
{
    ALPHA_SPMAT_BSR *conjugated_mat;
    transpose_conj_bsr(matA, &conjugated_mat);
    alphasparse_status_t status = spmmd_bsr_col_plain(conjugated_mat,matB,matC,ldc);
    destroy_bsr(conjugated_mat);
    return status;
}
