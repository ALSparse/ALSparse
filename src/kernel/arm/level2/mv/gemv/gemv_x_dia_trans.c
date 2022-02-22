#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static alphasparse_status_t ONAME_omp(const ALPHA_Number alpha,
		             const ALPHA_SPMAT_DIA* A,
		             const ALPHA_Number* x,
		             const ALPHA_Number beta,
		             ALPHA_Number* y)
{
    const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;
	const ALPHA_INT thread_num = alpha_get_thread_num();

	ALPHA_Number** tmp = (ALPHA_Number**)malloc(sizeof(ALPHA_Number*) * thread_num);
	for(int i = 0; i < thread_num; ++i)
	{
		tmp[i] = malloc(sizeof(ALPHA_Number) * n);
		memset(tmp[i], 0, sizeof(ALPHA_Number) * n);
	}
	const ALPHA_INT diags = A->ndiag;
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
    for (ALPHA_INT i = 0; i < diags; ++i)
    {
		const ALPHA_INT threadId = alpha_get_thread_id();
		const ALPHA_INT dis = A->distance[i];
        const ALPHA_INT row_start = alpha_max(0, -dis);
		const ALPHA_INT col_start = alpha_max(0, dis);
		const ALPHA_INT nnz = (m - row_start)<(n - col_start)?(m - row_start):(n - col_start);
		const ALPHA_INT start = i * A->lval;
		for (ALPHA_INT j = 0; j < nnz; ++j)
		{
			ALPHA_Number v;
			alpha_mul(v, alpha, A->values[start + row_start + j]);
			alpha_madde(tmp[threadId][col_start + j], v, x[row_start + j]);
		}
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
	for(ALPHA_INT i = 0; i < n; ++i)
	{ 
		alpha_mul(y[i], y[i], beta);
		for(ALPHA_INT j = 0; j < thread_num; ++j)
	  	{
			alpha_add(y[i], y[i], tmp[j][i]);
	 	}
	}  

#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
	for(int i = 0; i < thread_num; ++i)
	{
		alpha_free(tmp[i]);
	}

	alpha_free(tmp);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}


alphasparse_status_t
ONAME(const ALPHA_Number alpha,
		         const ALPHA_SPMAT_DIA* A,
		         const ALPHA_Number* x,
		         const ALPHA_Number beta,
		         ALPHA_Number* y)
{
	return ONAME_omp(alpha, A, x, beta, y);
}
