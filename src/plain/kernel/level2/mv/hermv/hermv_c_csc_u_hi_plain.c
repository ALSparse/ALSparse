#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

alphasparse_status_t
ONAME(const ALPHA_Complex alpha,
      const ALPHA_SPMAT_CSC *A,
      const ALPHA_Complex *x,
      const ALPHA_Complex beta,
      ALPHA_Complex *y)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
    

    for(ALPHA_INT i = 0; i < m; ++i)
	{
        ALPHA_Complex tmp1, tmp2;
        alpha_mul(tmp1, alpha, x[i]); 
        alpha_mul(tmp2, beta, y[i]); 
        alpha_add(y[i], tmp1, tmp2);
		// y[i] = beta * y[i] + alpha * x[i];
	}
	for(ALPHA_INT i = 0; i < n; ++i)
    {
        for(ALPHA_INT ai = A->cols_start[i]; ai < A->cols_end[i]; ++ai)
        {
            const ALPHA_INT row = A->row_indx[ai];
            if(row >= i)
            {                                                                           
                continue;
            }
            else
            {   
                ALPHA_Complex tmp;
                ALPHA_Complex conval;
				conval.real = A->values[ai].real;
				conval.imag = 0.0 - A->values[ai].imag;

                alpha_mul(tmp, A->values[ai], x[i]); 
            	alpha_mul(tmp, alpha, tmp); 
            	alpha_add(y[row], y[row], tmp);
				
				alpha_mul(tmp, conval, x[row]); 
            	alpha_mul(tmp, alpha, tmp); 
            	alpha_add(y[i], y[i], tmp);
 
                // y[col] += alpha * A->values[ai] * x[i];
                // y[i] += alpha * A->values[ai] * x[col];
            }
        }
    }
    
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
