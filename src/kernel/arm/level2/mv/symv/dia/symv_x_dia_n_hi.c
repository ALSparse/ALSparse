#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
		              const ALPHA_SPMAT_DIA *A,
		              const ALPHA_Number *x,
		              const ALPHA_Number beta,
		              ALPHA_Number *y)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
    if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

    for(ALPHA_INT i = 0; i < m; ++i)
	{
		alpha_mul(y[i], beta, y[i]);
	}
	const ALPHA_INT diags = A->ndiag;
	for(ALPHA_INT i = 0; i < diags; ++i)
    {
		const ALPHA_INT dis = A->distance[i];
		if(dis == 0)
		{
			const ALPHA_INT start = i * A->lval;
			for(ALPHA_INT j = 0; j < m; ++j)
			{
				ALPHA_Number v;
				alpha_mul(v, alpha, A->values[start + j]);
				alpha_madde(y[j], v, x[j]);
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
				ALPHA_Number v;
				alpha_mul(v, alpha, A->values[start + j]);
				alpha_madde(y[row_start + j], v, x[col_start + j]);
				alpha_madde(y[col_start + j], v, x[row_start + j]);
			}
		}
    }
    
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
