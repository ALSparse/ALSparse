#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
static alphasparse_status_t
symv_csc_u_lo_serial(const ALPHA_Number alpha,
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
        ALPHA_Number tmp1, tmp2;
        alpha_mul(tmp1, alpha, x[i]); 
        alpha_mul(tmp2, beta, y[i]); 
        alpha_add(y[i], tmp1, tmp2);
    }
    
    for(ALPHA_INT i = 0; i < n; ++i)
    {
        for(ALPHA_INT ai = A->cols_start[i]; ai < A->cols_end[i]; ++ai)
        {
            const ALPHA_INT row = A->row_indx[ai];
            if(row > i)
            {                
                ALPHA_Number tmp;
                alpha_conj(tmp, A->values[ai]);
                alpha_mule(tmp, x[i]); 
            	alpha_mul(tmp, alpha, tmp); 
            	alpha_add(y[row], y[row], tmp);
				
				alpha_conj(tmp, A->values[ai]);
                alpha_mule(tmp, x[row]); 
            	alpha_mul(tmp, alpha, tmp); 
            	alpha_add(y[i], y[i], tmp);                                                            
            }                                                                           
        }
    }

	return ALPHA_SPARSE_STATUS_SUCCESS;
}
alphasparse_status_t
ONAME(const ALPHA_Number alpha,
		              const ALPHA_SPMAT_CSC *A,
		              const ALPHA_Number *x,
		              const ALPHA_Number beta,
		              ALPHA_Number *y)
{
	return symv_csc_u_lo_serial(alpha, A, x, beta, y);
}
