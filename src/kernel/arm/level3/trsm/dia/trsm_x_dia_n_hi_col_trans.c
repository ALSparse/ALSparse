#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_DIA *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_DIA *transposed_mat;
    transpose_dia(A, &transposed_mat);
    alphasparse_status_t status = trsm_dia_n_lo_col(alpha, transposed_mat, x, columns, ldx, y, ldy);
    destroy_dia(transposed_mat);
    return status;
}
