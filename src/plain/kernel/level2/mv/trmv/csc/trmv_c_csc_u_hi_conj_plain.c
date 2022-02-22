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
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
    if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

    for(ALPHA_INT i = 0;i < m; ++i)
    {
        ALPHA_Number tmp;
        tmp = x[i];
        for(ALPHA_INT ai = A->cols_start[i]; ai < A->cols_end[i]; ++ai)
        {
            const ALPHA_INT row = A->row_indx[ai];
            if(row < i)
            {
                ALPHA_Number tmp1;
                alpha_conj(tmp1, A->values[ai]);
                alpha_mule(tmp1, x[row]); 
                alpha_add(tmp, tmp1, tmp);
                // tmp += A->values[ai] * x[row];
            }
        }
        alpha_mul(tmp, tmp, alpha); 
        ALPHA_Number tmp1;
        alpha_mul(tmp1, beta, y[i]); 
        alpha_add(y[i], tmp1, tmp);
        // y[i] = beta * y[i] + alpha * tmp;                                               
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
