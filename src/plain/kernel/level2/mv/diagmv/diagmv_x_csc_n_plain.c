#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

#include <stdio.h>
alphasparse_status_t
ONAME(const ALPHA_Number alpha,
                     const ALPHA_SPMAT_CSC *A,
                     const ALPHA_Number *x,
                     const ALPHA_Number beta,
                     ALPHA_Number *y)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
    if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    ALPHA_INT flag = 1;
    for(ALPHA_INT i = 0; i < n; ++i)
    {
        ALPHA_Number t;
        alpha_mul(t, y[i], beta);
        if(flag) 
        {
            y[i] = t;
        }
        for(ALPHA_INT ai = A->cols_start[i]; ai < A->cols_end[i]; ++ai)
        {
            const ALPHA_INT row = A->row_indx[ai];
            if(i == row)
            {
                ALPHA_Number tmp;
                if(flag)
                {
                    alpha_mul(tmp, A->values[ai], x[row]);     
                    alpha_mul(tmp, alpha, tmp);   
                    alpha_add(y[i], y[i], tmp);  
                    // y[i] += alpha * A->values[ai] * x[col];
                    flag = 0;
                }
                else
                {
                    alpha_mul(tmp, A->values[ai], x[row]);     
                    alpha_mul(tmp, alpha, tmp); 
                    alpha_add(y[i], t, tmp);  
                    // y[i] = alpha * A->values[ai] * x[col] + t;
                }
                break;
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
