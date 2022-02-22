#include "alphasparse/opt.h"
#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{//assume A is square
    ALPHA_Number* diag=(ALPHA_Number*) alpha_malloc(A->rows*sizeof(ALPHA_Number));

    memset(diag, '\0', A->rows * sizeof(ALPHA_Number));
    ALPHA_INT num_thread = alpha_get_thread_num(); 
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for (ALPHA_INT c = 0; c < A->cols; c++)
    {
        for (ALPHA_INT ai = A->cols_start[c]; ai < A->cols_end[c]; ai++)
        {
            ALPHA_INT ar = A->row_indx[ai];
            if (ar == c)
            {
                diag[c] = A->values[ai];
            }
        }
    }

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for (ALPHA_INT r = 0; r < A->rows; ++r)
    {
        for (ALPHA_INT c = 0; c < columns; ++c)
        {
            ALPHA_Number t;
            alpha_mul(t, alpha, x[index2(r, c, ldx)]);
            alpha_div(y[index2(r, c, ldy)], t, diag[r]);
        }
    }
    alpha_free(diag);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
