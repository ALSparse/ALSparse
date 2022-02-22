#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <memory.h>
#include <stdlib.h>

alphasparse_status_t
ONAME(const ALPHA_Complex alpha,
		              const ALPHA_SPMAT_DIA *A,
		              const ALPHA_Complex *x,
		              const ALPHA_Complex beta,
		              ALPHA_Complex *y)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
    if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

	const ALPHA_INT thread_num = alpha_get_thread_num();
	ALPHA_Number** tmp = (ALPHA_Number**)malloc(sizeof(ALPHA_Number*) * thread_num);
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
	for(int i = 0; i < thread_num; ++i)
	{
		tmp[i] = malloc(sizeof(ALPHA_Number) * m);
		memset(tmp[i], 0, sizeof(ALPHA_Number) * m);
	} 

	const ALPHA_INT diags = A->ndiag;
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
    for (ALPHA_INT i = 0; i < diags; ++i)
    {
		const ALPHA_INT threadId = alpha_get_thread_id();
		const ALPHA_INT dis = A->distance[i];
		if(dis == 0)
		{
			const ALPHA_INT start = i * A->lval;
			for(ALPHA_INT j = 0; j < m; ++j)
			{
				ALPHA_Number v;
				alpha_mul_3c(v, alpha, A->values[start + j]);
				alpha_madde(tmp[threadId][j], v, x[j]);
			}
		}
		else if(dis > 0)
		{ 
			const ALPHA_INT row_start = 0;
			const ALPHA_INT col_start = dis;
			const ALPHA_INT nnz = m - dis;
			const ALPHA_INT start = i * A->lval;
			for(ALPHA_INT j = 0; j < nnz; ++j)
			{
				ALPHA_Complex v,v_c;
				ALPHA_Complex val_orig = A->values[start + j];
				ALPHA_Complex val_conj = {A->values[start + j].real,-A->values[start + j].imag};
				alpha_mul(v, alpha, val_orig);
				alpha_mul(v_c, alpha, val_conj);
				alpha_madde(tmp[threadId][col_start + j], v, x[row_start + j]);
				alpha_madde(tmp[threadId][row_start + j], v_c, x[col_start + j]);
			}
		}
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
	for(ALPHA_INT i = 0; i < m; ++i)
	{
		alpha_mul(y[i], beta, y[i]);
		for(ALPHA_INT j = 0; j < thread_num; ++j)
		{
			alpha_add(y[i], y[i], tmp[j][i]);
	 	}
	} 
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
    for (ALPHA_INT i = 0; i < thread_num; ++i)
    {
        alpha_free(tmp[i]);
    }
    alpha_free(tmp);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
