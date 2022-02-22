#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

static alphasparse_status_t
hermm_coo_n_hi_col_transXY(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT ldX = columns;
    ALPHA_INT ldY = columns;
    ALPHA_Number *X_ = alpha_malloc(mat->cols * ldX * sizeof(ALPHA_Number));
    ALPHA_Number *Y_ = alpha_malloc(mat->rows * ldY * sizeof(ALPHA_Number));

    pack_matrix_col2row(mat->cols, columns, x, ldx, X_, ldX);
    pack_matrix_col2row(mat->rows, columns, y, ldy, Y_, ldY);
    hermm_coo_n_hi_row(alpha, mat, X_, columns, ldX, beta, Y_, ldY); 

    pack_matrix_row2col(mat->rows, columns, Y_, ldY, y, ldy);
    alpha_free(X_);
    alpha_free(Y_);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparse_status_t
ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    return hermm_coo_n_hi_col_transXY(alpha, mat, x, columns, ldx, beta, y, ldy);
}