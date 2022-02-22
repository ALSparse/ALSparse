#include "alphasparse/kernel.h"
#include "alphasparse/kernel_plain.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static alphasparse_status_t
symv_coo_u_lo_omp(const ALPHA_Number alpha,
				  const ALPHA_SPMAT_COO *A,
				  const ALPHA_Number *x,
				  const ALPHA_Number beta,
				  ALPHA_Number *y)
{
	const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;

	const ALPHA_INT nnz = A->nnz;
	const ALPHA_INT thread_num = alpha_get_thread_num();

	ALPHA_Number **tmp = (ALPHA_Number **)malloc(sizeof(ALPHA_Number *) * thread_num);
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
	for (int i = 0; i < thread_num; ++i)
	{
		tmp[i] = malloc(sizeof(ALPHA_Number) * m);
		memset(tmp[i], 0, sizeof(ALPHA_Number) * m);
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
	for (ALPHA_INT i = 0; i < nnz; i++)
	{
		const ALPHA_INT threadId = alpha_get_thread_id();
		const ALPHA_INT r = A->row_indx[i];
		const ALPHA_INT c = A->col_indx[i];
		if (r <= c)
		{
			continue;
		}
		ALPHA_Number v;
		alpha_mul_3c(v, alpha, A->values[i]);
		alpha_madde(tmp[threadId][r], v, x[c]);
		alpha_madde(tmp[threadId][c], v, x[r]);
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
	for (ALPHA_INT i = 0; i < m; ++i)
	{
		alpha_mul(y[i], beta, y[i]);
		alpha_madde(y[i], alpha, x[i]);
		for (ALPHA_INT j = 0; j < thread_num; ++j)
		{
			alpha_add(y[i], y[i], tmp[j][i]);
		}
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
	for (int i = 0; i < thread_num; ++i)
	{
		alpha_free(tmp[i]);
	}
	alpha_free(tmp);
	return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
	  const ALPHA_SPMAT_COO *A,
	  const ALPHA_Number *x,
	  const ALPHA_Number beta,
	  ALPHA_Number *y)
{
	const ALPHA_INT thread_num = alpha_get_thread_num();
	return symv_coo_u_lo_omp(alpha, A, x, beta, y);
}
