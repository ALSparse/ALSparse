#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include <memory.h>
#ifdef _OPENMP
#include <omp.h>
#endif

alphasparse_status_t ONAME(const ALPHA_SPMAT_CSC *matA, const ALPHA_SPMAT_CSC *matB, ALPHA_Number *matC, const ALPHA_INT ldc)
{
    if (matA->cols != matB->rows)
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;

    ALPHA_INT m = matA->rows;
    ALPHA_INT n = matB->cols;
    ALPHA_INT num_thread = alpha_get_thread_num();

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for(ALPHA_INT i = 0; i < matA->rows; i++)
    {
        for(ALPHA_INT j = 0; j < matB->cols; j++)
        {
            alpha_setzero(matC[index2(i, j, ldc)]);
        }
    }

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for (ALPHA_INT bc = 0; bc < n; bc++)
    {
        for (ALPHA_INT bi = matB->cols_start[bc]; bi < matB->cols_end[bc]; bi++)
        {
            ALPHA_INT ac = matB->row_indx[bi]; // ac == br
            ALPHA_Number bv;
            bv = matB->values[bi];
            for (ALPHA_INT ai = matA->cols_start[ac]; ai < matA->cols_end[ac]; ai++)
            {
                ALPHA_INT ar = matA->row_indx[ai];
                ALPHA_Number av;
                av = matA->values[ai];
                ALPHA_Number tmp;
                alpha_mul(tmp, av, bv);
                alpha_adde(matC[index2(ar, bc, ldc)], tmp);
            }
        }
    }
    
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
