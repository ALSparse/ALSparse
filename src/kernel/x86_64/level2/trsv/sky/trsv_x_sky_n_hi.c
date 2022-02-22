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

    for (ALPHA_INT c = A->cols - 1; c >= 0; c--)
    {
        ALPHA_Number temp;
        alpha_setzero(temp);
        for (ALPHA_INT ic = A->cols - 1; ic > c; ic--)
        {
            ALPHA_INT start = A->pointers[ic];
            ALPHA_INT end   = A->pointers[ic + 1];
            ALPHA_INT eles_num = ic - c;
            if(end - eles_num - 1 >= start)
                alpha_madde(temp, A->values[end - eles_num - 1], y[ic]);
        }

        ALPHA_Number t;
        alpha_mul(t, alpha, x[c]);
        alpha_sub(t, t, temp);
        alpha_div(y[c], t, diag[c]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
