#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif

alphasparse_status_t
ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat_, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_COO *transposed_mat;
    transpose_coo(mat_, &transposed_mat);
    gemm_coo_row(alpha, transposed_mat, x, columns, ldx, beta, y, ldy);
    destroy_coo(transposed_mat);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
