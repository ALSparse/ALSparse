#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_SPMAT_CSR *matA, const ALPHA_SPMAT_CSR *matB, ALPHA_Number *matC, const ALPHA_INT ldc)
{
    if (matA->cols != matB->rows || ldc < matA->rows)
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;

    ALPHA_INT m = matA->rows;

    for(ALPHA_INT i = 0; i < matA->rows; i++)
        for(ALPHA_INT j = 0; j < matB->cols; j++)
        {
            alpha_setzero(matC[index2(j, i, ldc)]);
        }
        
    ALPHA_INT num_thread = alpha_get_thread_num();
    ALPHA_INT64 *flop = (ALPHA_INT64 *)alpha_malloc(matB->cols * sizeof(ALPHA_INT64));
    memset(flop, '\0', m * sizeof(ALPHA_INT64));

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for (ALPHA_INT ar = 0; ar < m; ar++)
    {
        for (ALPHA_INT ai = matA->rows_start[ar]; ai < matA->rows_end[ar]; ai++)
        {
            ALPHA_INT br = matA->col_indx[ai];
            flop[ar] += matB->rows_end[br] - matB->rows_start[br];
        }
    }
    for (ALPHA_INT i = 1; i < m; i++)
    {
        flop[i] += flop[i - 1];
    }
    ALPHA_INT partition[num_thread + 1];
    balanced_partition_row_by_flop(flop, m, num_thread, partition);

#ifdef _OPENMP
#pragma omp parallel num_threads(num_thread)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();
        ALPHA_INT local_m_s = partition[tid];
        ALPHA_INT local_m_e = partition[tid + 1];
        for (ALPHA_INT ar = local_m_s; ar < local_m_e; ar++)
        {
            for (ALPHA_INT ai = matA->rows_start[ar]; ai < matA->rows_end[ar]; ai++)
            {
                ALPHA_INT br = matA->col_indx[ai];
                ALPHA_Number av = matA->values[ai];
                for (ALPHA_INT bi = matB->rows_start[br]; bi < matB->rows_end[br]; bi++)
                {
                    ALPHA_INT bc = matB->col_indx[bi];
                    ALPHA_Number bv = matB->values[bi];
                    alpha_madde(matC[index2(bc,ar,ldc)], av, bv);
                }
            }
        }
    }
    alpha_free(flop);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
