#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"                                                                    
#ifdef _OPENMP
#include <omp.h>
#endif

static alphasparse_status_t 
diagmv_csc_u_omp(const ALPHA_Number alpha,
		       const ALPHA_SPMAT_CSC *A,
		       const ALPHA_Number *x,
		       const ALPHA_Number beta,
		       ALPHA_Number *y)
{
	const ALPHA_INT n = A->cols;
	const ALPHA_INT thread_num = alpha_get_thread_num();
 
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif

    for(ALPHA_INT i = 0; i < n; ++i)
    {
	    ALPHA_Number tmp;
        alpha_mul(tmp, alpha, x[i]);  
        alpha_mul(y[i], y[i], beta);  
        alpha_add(y[i], y[i], tmp); 
        // y[i] = beta * y[i] + alpha * x[i];                                              
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
static alphasparse_status_t
diagmv_csc_u_serial(const ALPHA_Number alpha,
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
alphasparse_status_t 
ONAME(const ALPHA_Number alpha,
		       const ALPHA_SPMAT_CSC *A,
		       const ALPHA_Number *x,
		       const ALPHA_Number beta,
		       ALPHA_Number *y)
{
    return diagmv_csc_u_serial(alpha, A, x, beta, y);
}
