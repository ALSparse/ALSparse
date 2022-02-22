#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "memory.h"
#include "alphasparse/opt.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT rowA = mat->rows;
    ALPHA_INT nnz = mat->nnz;
    ALPHA_INT num_threads = alpha_get_thread_num();

    ALPHA_Number diag[rowA];
    memset(diag, '\0', sizeof(ALPHA_Number) * rowA);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT ar = 0; ar < nnz; ++ar)
    {
        if (mat->col_indx[ar] == mat->row_indx[ar])
        {
            diag[mat->row_indx[ar]] = mat->values[ar];
        }
    }

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
        for (ALPHA_INT cr = 0; cr < rowA; ++cr)
        {
            ALPHA_Number t;
            alpha_setzero(t);
            alpha_mul(t, alpha, diag[cr]);
            alpha_mul(t, t, x[index2(cc, cr, ldx)]);
            alpha_mul(y[index2(cc, cr, ldy)], beta, y[index2(cc, cr, ldy)]);
            alpha_add(y[index2(cc, cr, ldy)], y[index2(cc, cr, ldy)], t);
        }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
