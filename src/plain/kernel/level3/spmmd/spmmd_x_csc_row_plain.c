#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_SPMAT_CSC *matA, const ALPHA_SPMAT_CSC *matB, ALPHA_Number *matC, const ALPHA_INT ldc)
{
    if (matA->cols != matB->rows)
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;

    ALPHA_INT m = matA->rows;
    ALPHA_INT n = matB->cols;

    //memset(matC, '\0', m * ldc * sizeof(ALPHA_Number));
    if(ldc == matB->cols)
    {
        memset(matC, '\0', matA->rows * ldc * sizeof(ALPHA_Number));
    }
    else
    {
        for(ALPHA_INT i = 0; i < matA->rows; i++)
        {
            for(ALPHA_INT j = 0; j < matB->cols; j++)
            {
                alpha_setzero(matC[index2(i, j, ldc)]);
            }
        }
    }

    // 计算
    for (ALPHA_INT bc = 0; bc < n; bc++)
    {
        for (ALPHA_INT bi = matB->cols_start[bc]; bi < matB->cols_end[bc]; bi++)
        {
            ALPHA_INT ac = matB->row_indx[bi]; // ac == br
            //ALPHA_Number bv = matB->values[bi]; // bv := B[br][bc]
            ALPHA_Number bv;
            bv = matB->values[bi];
            for (ALPHA_INT ai = matA->cols_start[ac]; ai < matA->cols_end[ac]; ai++)
            {
                ALPHA_INT ar = matA->row_indx[ai];
                //ALPHA_Number av = matA->values[ai];
                //matC[index2(ar, bc, ldc)] += av * bv;
                ALPHA_Number av;
                av = matA->values[ai];
                ALPHA_Number tmp;
                alpha_mul(tmp, av, bv);
                alpha_adde(matC[index2(ar, bc, ldc)], tmp);
                // matC[index2(ar, bc, ldc)].real += tmp.real;
                // matC[index2(ar, bc, ldc)].imag += tmp.imag;
            }
        }
    }
    
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
