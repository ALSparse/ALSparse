#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat_, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_COO *mat = alpha_malloc(sizeof(ALPHA_SPMAT_COO));
    mat -> values = mat_->values;
    mat -> row_indx = mat_->col_indx;
    mat -> col_indx = mat_->row_indx;
    mat -> rows = mat_->cols;
    mat -> cols = mat_->rows;
    mat -> nnz = mat_->nnz;
    
    alphasparse_status_t status = trmm_coo_u_lo_row(alpha, mat, x, columns, ldx, beta, y, ldy);
    alpha_free(mat);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
