#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat_, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    // ALPHA_SPMAT_COO *transposed_mat;
    // transpose_coo(mat, &transposed_mat);
    // alphasparse_status_t status = trmm_coo_n_lo_row(alpha, transposed_mat, x, columns, ldx, beta, y, ldy);
    // destroy_coo(transposed_mat);
    // return status;
    // printf("trmm_coo_n_hi_row_trans\n");
    ALPHA_SPMAT_COO *mat = alpha_malloc(sizeof(ALPHA_SPMAT_COO));
    mat->values = mat_->values;
    mat->row_indx = mat_->col_indx;
    mat->col_indx = mat_->row_indx;
    mat->rows = mat_->cols;
    mat->cols = mat_->rows;
    mat->nnz = mat_->nnz;

    alphasparse_status_t status = trmm_coo_n_lo_row(alpha, mat, x, columns, ldx, beta, y, ldy);

    alpha_free(mat);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
