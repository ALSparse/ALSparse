#include "alphasparse/kernel_plain.h"
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
		alpha_madde(y[i], alpha, x[i]);
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;
 }
