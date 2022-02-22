#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_SKY *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT rowC = mat->rows;
    ALPHA_INT colC = columns;
    ALPHA_INT num_threads = alpha_get_thread_num();

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT c = 0; c < colC; ++c)
    {
        for (ALPHA_INT r = 0; r < rowC; ++r)
        {
            ALPHA_Number t;
            alpha_mul(t, alpha, x[index2(c, r, ldx)]);
            alpha_mul(y[index2(c, r, ldy)], beta, y[index2(c, r, ldy)]);
            alpha_add(y[index2(c, r, ldy)], y[index2(c, r, ldy)], t);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
