#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

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

    for(ALPHA_INT i = 0; i < m; ++i)
	{
		alpha_mul(y[i], beta, y[i]);
		alpha_madde(y[i], alpha, x[i]);
	}
	const ALPHA_INT diags = A->ndiag;
	for(ALPHA_INT i = 0; i < diags; ++i)
    {
		const ALPHA_INT dis = A->distance[i];
		if(dis > 0)
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
				alpha_madde(y[row_start + j], v, x[col_start + j]);
				alpha_madde(y[col_start + j], v_c, x[row_start + j]);
			}
		}
    }
    
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
