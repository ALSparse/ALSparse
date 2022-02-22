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

    for(ALPHA_INT i = 0; i < n; ++i)
    {
        // y[i] *= beta;
        alpha_mul(y[i], y[i], beta); 
        for(ALPHA_INT ai = A->cols_start[i]; ai < A->cols_end[i]; ++ai)
        {
            ALPHA_Number tmp;
			alpha_setzero(tmp); 
            const ALPHA_INT row = A->row_indx[ai];
            if(row <= i)
            {
                alpha_mul(tmp, A->values[ai], x[i]); 
				alpha_mul(tmp, alpha, tmp);
        		alpha_add(y[row], tmp, y[row]);
                // y[row] += alpha * A->values[ai] * x[i];
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
