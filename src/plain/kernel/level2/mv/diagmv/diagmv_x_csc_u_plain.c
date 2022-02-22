#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
                     const ALPHA_SPMAT_CSC *A,
                     const ALPHA_Number *x,
                     const ALPHA_Number beta,
                     ALPHA_Number *y)
{                                                                          
    const int m = A->cols;

    for(int i = 0; i < m; ++i)
    {
        ALPHA_Number tmp;
        alpha_mul(tmp, alpha, x[i]);  
        alpha_mul(y[i], y[i], beta);  
        alpha_add(y[i], y[i], tmp);   
        // y[i] = beta * y[i] + alpha * x[i];
    }  
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
