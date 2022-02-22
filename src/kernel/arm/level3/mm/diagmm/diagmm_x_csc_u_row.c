#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT rowC = mat->rows;
    ALPHA_INT colC = columns;
    ALPHA_INT num_threads = alpha_get_thread_num();

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT r = 0; r < rowC; ++r)
    {
        for (ALPHA_INT c = 0; c < colC; ++c)
        {
            ALPHA_Number temp1, temp2;
            alpha_mul(temp1, beta, y[index2(r, c, ldy)]);
            alpha_mul(temp2, alpha, x[index2(r, c, ldx)]);
            alpha_add(y[index2(r, c, ldy)], temp1, temp2);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
