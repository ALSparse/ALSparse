#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif

alphasparse_status_t ONAME(const ALPHA_SPMAT_BSR *matA, const ALPHA_SPMAT_BSR *matB, ALPHA_Number *matC, const ALPHA_INT ldc)
{
    ALPHA_SPMAT_BSR *transposed_mat;
    transpose_bsr(matA, &transposed_mat);
    alphasparse_status_t status = spmmd_bsr_col(transposed_mat,matB,matC,ldc);
    destroy_bsr(transposed_mat);
    return status;
}
