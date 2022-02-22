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
    
    for (ALPHA_INT c = 0; c < A->cols; c++)
    {
        alpha_mul(y[c], alpha, x[c]);
    }
    for (ALPHA_INT ac = A->cols - 1; ac >= 0; ac--) 
    {
        alpha_div(y[ac], y[ac], diag[ac]);
	    for (ALPHA_INT ai = A->cols_start[ac]; ai < A->cols_end[ac]; ai++) 
        {
            ALPHA_INT ar = A->row_indx[ai];
            ALPHA_Number val;
            val = A->values[ai];
            if (ac > ar) 
            {
                ALPHA_Number t;
                alpha_mul(t, val, y[ac]);
                alpha_sub(y[ar], y[ar], t);
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
