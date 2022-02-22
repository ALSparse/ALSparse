#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif

alphasparse_status_t ONAME(const ALPHA_SPMAT_BSR *matA, const ALPHA_SPMAT_BSR *matB, ALPHA_Number *matC, const ALPHA_INT ldc)
{
    ALPHA_SPMAT_BSR *conjugated_mat;
    transpose_conj_bsr(matA, &conjugated_mat);
    alphasparse_status_t status = spmmd_bsr_col(conjugated_mat,matB,matC,ldc);
    destroy_bsr(conjugated_mat);
    return status;
}
