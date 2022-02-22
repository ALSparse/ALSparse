#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif

alphasparse_status_t
ONAME(const ALPHA_Complex alpha,
	  const ALPHA_SPMAT_CSR *A,
	  const ALPHA_Complex *x,
	  const ALPHA_Complex beta,
	  ALPHA_Complex *y)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
    
    ALPHA_INT num_threads = alpha_get_thread_num();

    for(ALPHA_INT i = 0; i < m; ++i)
    {
        alpha_mul(y[i], y[i], beta); 
        for(ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i]; ++ai)
        {
            const ALPHA_INT col = A->col_indx[ai];
            ALPHA_Complex tmp;
			tmp.real = 0.0;
			tmp.imag = 0.0;  
            if(col < i)
            {
                ALPHA_Complex conval;
				conval.real = A->values[ai].real;
				conval.imag = 0.0 - A->values[ai].imag;
                alpha_mul(tmp, conval, x[i]); 
            	alpha_mul(tmp, alpha, tmp); 
            	alpha_add(y[col], y[col], tmp);
				
                alpha_mul(tmp, A->values[ai], x[col]); 
            	alpha_mul(tmp, alpha, tmp); 
            	alpha_add(y[i], y[i], tmp);                             
            }
			else if(col == i)
			{
                alpha_mul(tmp, A->values[ai], x[col]); 
            	alpha_mul(tmp, alpha, tmp); 
            	alpha_add(y[i], y[i], tmp);
			}
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
