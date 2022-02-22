#include <string.h>

#include "alphasparse/opt.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include <stdio.h>

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
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
	for (ALPHA_INT i = 0; i < nnz; ++i)
	{
		const ALPHA_INT tid = alpha_get_thread_id();
		const ALPHA_INT r = A->row_indx[i];
		const ALPHA_INT c = A->col_indx[i];
		const ALPHA_Number origin_val = A->values[i];
		ALPHA_Number conj_val;
		alpha_conj(conj_val, origin_val);
		if (r <= c)
		{
			continue;
		}
		ALPHA_Number v, v_c;
		alpha_mul(v, origin_val, alpha);
		alpha_mul(v_c, conj_val, alpha);
		{
			alpha_madde(tmp[tid][r], v, x[c]);
			alpha_madde(tmp[tid][c], v_c, x[r]);
		}
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
	return ALPHA_SPARSE_STATUS_SUCCESS;
}