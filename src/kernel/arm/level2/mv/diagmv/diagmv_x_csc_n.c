#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdio.h>
static alphasparse_status_t
diagmv_csc_n_omp(const ALPHA_Number alpha,
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
    for (ALPHA_INT i = 0; i < n; ++i)
    {
	    register ALPHA_Number tmp;
        alpha_setzero(tmp);
        for (ALPHA_INT ai = A->cols_start[i]; ai < A->cols_end[i]; ++ai)
        {
            if (A->row_indx[ai] == i)
            {
                alpha_mul(tmp, A->values[ai], x[i]);     
                alpha_mul(tmp, tmp, alpha);
                // tmp = alpha * A->values[ai] * x[i];
                break;
            }
        }
        alpha_mul(y[i], beta, y[i]);
        alpha_add(y[i], y[i], tmp);
        // y[i] = beta * y[i] + tmp;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

static alphasparse_status_t
diagmv_csc_n_serial(const ALPHA_Number alpha,
		             const ALPHA_SPMAT_CSC *A,
		             const ALPHA_Number *x,
		             const ALPHA_Number beta,
		             ALPHA_Number *y)
{
    const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;
	if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

	const ALPHA_INT thread_num = alpha_get_thread_num();
 
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
    for(ALPHA_INT i = 0; i < n; ++i)
	{
		ALPHA_Number t;
        alpha_mul(y[i], y[i], beta);

		for(ALPHA_INT ai = A->cols_start[i]; ai < A->cols_end[i]; ++ai)
		{
			const ALPHA_INT row = A->row_indx[ai];
			if(i == row)
			{
                ALPHA_Number tmp;
                alpha_mul(tmp, A->values[ai], x[row]);     
                alpha_mul(tmp, alpha, tmp);   
                alpha_add(y[i], y[i], tmp);  
                // y[i] += alpha * A->values[ai] * x[col];
				break;
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
    return diagmv_csc_n_serial(alpha, A, x, beta, y);
}
