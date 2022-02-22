#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static void
mm_coo_plain_outcols(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy, ALPHA_INT lrs, ALPHA_INT lre)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT tid = alpha_get_thread_id();
    for (ALPHA_INT nn = lrs; nn < lre; ++nn)
    {
        ALPHA_INT cr = mat->row_indx[nn];
        if (cr % num_threads != tid)
            continue; 

        ALPHA_Number *Y = &y[index2(cr, 0, ldy)];

        ALPHA_Number val;
        alpha_mul(val, alpha, mat->values[nn]);
        const ALPHA_Number *X = &x[index2(mat->col_indx[nn], 0, ldx)];
        ALPHA_INT c = 0;
        for (; c < columns; c++)
        {
            alpha_madde(Y[c], val, X[c]);
        }
    }
}

static alphasparse_status_t
mm_coo_omp(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT num_threads = alpha_get_thread_num();

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (int i = 0; i < mat->rows; i++)
        for (int j = 0; j < columns; j++)
            alpha_mul(y[index2(i, j, ldy)], y[index2(i, j, ldy)], beta);

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        mm_coo_plain_outcols(alpha, mat, x, columns, ldx, beta, y, ldy, 0, mat->nnz);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparse_status_t
ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    return mm_coo_omp(alpha, mat, x, columns, ldx, beta, y, ldy);
}
