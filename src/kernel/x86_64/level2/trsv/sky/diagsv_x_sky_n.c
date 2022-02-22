#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include <memory.h>
#ifdef _OPENMP
#include <omp.h>
#endif

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_SKY *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    ALPHA_Number diag[A->rows];

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

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif     
    for (ALPHA_INT r = 0; r < A->rows; ++r)
    {
        ALPHA_Number t;
        alpha_mul(t, alpha, x[r]);
        alpha_div(y[r], t, diag[r]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
