#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include <memory.h>
#ifdef _OPENMP
#include <omp.h>
#endif

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_SKY *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    ALPHA_Number diag[A->cols];
    memset(diag, '\0', A->rows * sizeof(ALPHA_Number));

    int num_thread = alpha_get_thread_num(); 

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif 
    for (ALPHA_INT r = 1; r < A->rows + 1; r++)
    {
        const ALPHA_INT indx = A->pointers[r] - 1;
        diag[r - 1] = A->values[indx];
    }

    for (ALPHA_INT r = 0; r <A->rows; r++)
    {
        ALPHA_Number temp;
        alpha_setzero(temp);

        ALPHA_INT start = A->pointers[r];
        ALPHA_INT end   = A->pointers[r + 1];
        ALPHA_INT idx = 1;
        ALPHA_INT eles_num = end - start;
        for (ALPHA_INT ai = start; ai < end - 1; ++ai)
        {
            ALPHA_INT c = r - eles_num + idx;
            alpha_madde(temp, A->values[ai], y[c]);
            idx ++;
        }     

        ALPHA_Number t;
        alpha_mul(t, alpha, x[r]);
        alpha_sub(t, t, temp);
        alpha_div(y[r], t, diag[r]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
