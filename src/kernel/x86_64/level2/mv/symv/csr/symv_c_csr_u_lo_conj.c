#include "alphasparse/kernel.h"                                                                 
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <memory.h>
#include<stdlib.h>

static alphasparse_status_t
symv_s_csr_u_lo_conj_omp(const ALPHA_Number alpha,
					const ALPHA_SPMAT_CSR *A,
					const ALPHA_Number *x,
					const ALPHA_Number beta,
					ALPHA_Number *y)
{
    const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;
	if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	ALPHA_INT num_threads = alpha_get_thread_num();

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
	for(ALPHA_INT i = 0; i < m; ++i)
	{
		alpha_mule(y[i], beta);
		alpha_madde(y[i], alpha, x[i]);
	}

	ALPHA_Number **y_local = alpha_memalign(num_threads * sizeof(ALPHA_Number *), DEFAULT_ALIGNMENT);

	for(ALPHA_INT i = 0; i < num_threads; i++)
	{
		y_local[i] = alpha_memalign(m * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);
		memset(y_local[i], '\0', sizeof(ALPHA_Number) * m);
	}
	
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(ALPHA_INT i = 0; i < m; ++i)
    {
		ALPHA_INT tid = alpha_get_thread_id();	
		ALPHA_Number tmp;	
		for(ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i]; ++ai)
		{
			const ALPHA_INT col = A->col_indx[ai];
			if(col >= i)
			{
			    continue;
			}
			else
			{	
				alpha_setzero(tmp);   
				cmp_conj(tmp, A->values[ai]);
                alpha_mul(tmp, alpha, tmp);
				alpha_madde(y_local[tid][col], tmp, x[i]);
				alpha_madde(y_local[tid][i], tmp, x[col]);       
			}
		}
    }

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
	for(ALPHA_INT row = 0; row < m; row++)
		for(ALPHA_INT i = 0; i < num_threads; i++)
			alpha_adde(y[row], y_local[i][row]);

	for(ALPHA_INT i = 0; i < num_threads; i++)
	{
		alpha_free(y_local[i]);
	}

	alpha_free(y_local);
    
	return ALPHA_SPARSE_STATUS_SUCCESS;
}


alphasparse_status_t
ONAME(const ALPHA_Number alpha,
	  const ALPHA_SPMAT_CSR *A,
	  const ALPHA_Number *x,
	  const ALPHA_Number beta,
	  ALPHA_Number *y)
{
    return symv_s_csr_u_lo_conj_omp(alpha, A, x, beta, y);
}
