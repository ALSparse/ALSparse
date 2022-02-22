#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    int num_threads = alpha_get_thread_num(); 
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT r = 0; r < A->rows; r++)
    {  
        alpha_mul(y[r], alpha, x[r]);      
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
