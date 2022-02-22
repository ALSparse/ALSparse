#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
		                    const ALPHA_SPMAT_DIA *A,
		                    const ALPHA_Number *x,
		                    const ALPHA_Number beta,
		                    ALPHA_Number *y)
{
#ifdef COMPLEX
    const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;
	if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	for(ALPHA_INT i = 0; i < m; ++i)
	{ 
		alpha_mul(y[i], beta, y[i]);
		alpha_madde(y[i], alpha, x[i]);
	}
	const ALPHA_INT diags = A->ndiag;
	for(ALPHA_INT i = 0; i < diags; ++i)
	{ 
		const ALPHA_INT dis = A->distance[i];
		if(dis < 0)
		{
			const ALPHA_INT row_start = -dis;
			const ALPHA_INT col_start = 0;
			const ALPHA_INT nnz = m + dis;
			const ALPHA_INT start = i * A->lval;
			for(ALPHA_INT j = 0; j < nnz; ++j)
			{
				ALPHA_Number v;
				alpha_mul_3c(v, alpha, A->values[start + row_start + j]);
				alpha_madde(y[col_start + j], v, x[row_start + j]);
	 	 	}
	 	} 
	}

	return ALPHA_SPARSE_STATUS_SUCCESS;
#else
	return ALPHA_SPARSE_STATUS_INVALID_VALUE;
#endif
}
