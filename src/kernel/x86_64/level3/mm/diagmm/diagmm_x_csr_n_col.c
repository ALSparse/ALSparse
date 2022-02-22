#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "memory.h"
#include "alphasparse/opt.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSR *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_Number *diag = alpha_malloc(mat->rows * sizeof(ALPHA_Number));
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT ar = 0; ar < mat->rows; ++ar)
    {
        alpha_setzero(diag[ar]);
        for (ALPHA_INT ai = mat->rows_start[ar]; ai < mat->rows_end[ar]; ++ai)
            if (mat->col_indx[ai] == ar)
            {
                diag[ar] = mat->values[ai];
            }
    }

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
        for (ALPHA_INT cr = 0; cr < mat->rows; ++cr)
        {
            ALPHA_Number val;
            alpha_mule(y[index2(cc, cr, ldy)], beta);
            alpha_mul(val, alpha, diag[cr]);
            alpha_madde(y[index2(cc, cr, ldy)], val, x[index2(cc, cr, ldx)]);
        }
    
    alpha_free(diag);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
