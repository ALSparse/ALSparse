#include "alphasparse/kernel_plain.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
	  const ALPHA_SPMAT_CSR *A,
	  const ALPHA_Number *x,
	  const ALPHA_Number beta,
	  ALPHA_Number *y)
{
    const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;
	for(ALPHA_INT i = 0; i < m; ++i)
	{
		alpha_mule(y[i], beta);
		for(ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i]; ++ai)
		{
			const ALPHA_INT col = A->col_indx[ai];
			if(i == col)
			{
				ALPHA_Number tmp;
        		alpha_setzero(tmp);
				alpha_mul(tmp, alpha, A->values[ai]);
				alpha_madde(y[i], tmp, x[col]);
				// y[i] += alpha * A->values[ai] * x[col];
				break;
			}
		}
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;
 }
