#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSR *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT cr = 0; cr < mat->rows; ++cr)
        {
            alpha_mule(y[index2(cc, cr, ldy)], beta);
            ALPHA_Number ctmp;
            alpha_setzero(ctmp);
            for (ALPHA_INT ai = mat->rows_start[cr]; ai < mat->rows_end[cr]; ++ai)
            {
                ALPHA_INT ac = mat->col_indx[ai];
                if (ac <= cr)
                {
                    alpha_madde(ctmp, mat->values[ai], x[index2(cc, ac, ldx)]);
                }
            }
            alpha_madde(y[index2(cc, cr, ldy)], ctmp, alpha);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
