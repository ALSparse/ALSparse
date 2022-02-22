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
    ALPHA_INT m = A->cols;
    for (ALPHA_INT r = 0; r < m; r++)
    {
        // y[r] *= beta;
        alpha_mul(y[r], y[r], beta); 
        ALPHA_Number tmp;
        alpha_setzero(tmp);        
        
        for (ALPHA_INT ai = A->cols_start[r]; ai < A->cols_end[r]; ai++)
        {            
            ALPHA_Number inner_tmp;
            alpha_setzero(inner_tmp);
            alpha_mul(inner_tmp, A->values[ai], x[A->row_indx[ai]]); 
            alpha_add(tmp, tmp, inner_tmp);
            // tmp += A->values[ai] * x[A->col_indx[ai]];
        }
        alpha_mul(tmp, alpha, tmp); 
        alpha_add(y[r], y[r], tmp); 
        // y[r] += alpha * tmp;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
