#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_COO *conjugated_mat;
    transpose_conj_coo(mat, &conjugated_mat);
    gemm_coo_col(alpha, conjugated_mat, x, columns, ldx, beta, y, ldy);
    destroy_coo(conjugated_mat);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
