#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    ALPHA_Number diag[A->cols];

    memset(diag, '\0', A->cols * sizeof(ALPHA_Number));

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
    ALPHA_Number tmp;
    for (ALPHA_INT r = 0; r < A->rows; ++r)
    {
        alpha_mul(tmp, alpha, x[r]); 
        alpha_div(y[r], tmp, diag[r]); 
    }
    
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
