#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_SKY *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = columns;
    ALPHA_INT num_threads = alpha_get_thread_num();

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT r = 0; r < m; ++r)
    {
        ALPHA_Number *Y = &y[index2(r, 0, ldy)];
        for (ALPHA_INT c = 0; c < n; c++)
            alpha_mul(Y[c], Y[c], beta);

        ALPHA_INT idx = mat->pointers[r + 1] - 1;
        ALPHA_Number val;
        alpha_mul(val, alpha, mat->values[idx]);
        const ALPHA_Number *X = &x[index2(r, 0, ldx)];
        for (ALPHA_INT c = 0; c < n; ++c)
            alpha_madde(Y[c], val, X[c]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
