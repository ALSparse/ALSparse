#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#define CACHELINE 64

static alphasparse_status_t
symm_coo_n_hi_col_transXY(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT ldX = columns;
    ALPHA_INT ldY = columns;
    ALPHA_Number *X_ = alpha_malloc(mat->cols * ldX * sizeof(ALPHA_Number));
    ALPHA_Number *Y_ = alpha_malloc(mat->rows * ldY * sizeof(ALPHA_Number));

    pack_matrix_col2row(mat->cols, columns, x, ldx, X_, ldX);
    pack_matrix_col2row(mat->rows, columns, y, ldy, Y_, ldY);
    symm_coo_n_hi_row_conj(alpha, mat, X_, columns, ldX, beta, Y_, ldY); //alpha,mat,X,columns,ldX,beta,Y,ldY

    pack_matrix_row2col(mat->rows, columns, Y_, ldY, y, ldy);
    alpha_free(X_);
    alpha_free(Y_);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparse_status_t
ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT block_size = CACHELINE / sizeof(ALPHA_Number);
    ALPHA_INT block_num = (columns + block_size - 1) / block_size;
    ALPHA_INT threads = num_threads;
    if (num_threads > block_num)
        threads = block_num;
    alpha_set_thread_num(threads);
    alphasparse_status_t st = symm_coo_n_hi_col_transXY(alpha, mat, x, columns, ldx, beta, y, ldy);
    alpha_set_thread_num(num_threads);
    return st;
}