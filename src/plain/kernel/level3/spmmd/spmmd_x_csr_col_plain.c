#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
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
    // 计算
    {
        for (ALPHA_INT ar = 0; ar < m; ar++)
        {
            for (ALPHA_INT ai = matA->rows_start[ar]; ai < matA->rows_end[ar]; ai++)
            {
                ALPHA_INT br = matA->col_indx[ai];
                ALPHA_Number av = matA->values[ai];
               for (ALPHA_INT bi = matB->rows_start[br]; bi < matB->rows_end[br]; bi++)
               {
                    ALPHA_INT bc = matB->col_indx[bi];
                    ALPHA_Number bv = matB->values[bi];
                    alpha_madde(matC[index2(bc, ar, ldc)], av, bv);
               }
            }
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
