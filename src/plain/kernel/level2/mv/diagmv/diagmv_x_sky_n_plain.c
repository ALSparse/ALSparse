#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
		             const ALPHA_SPMAT_SKY *A,
		             const ALPHA_Number *x,
		             const ALPHA_Number beta,
		             ALPHA_Number *y)
{
    const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;
	if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

	for(ALPHA_INT i = 0; i < m; ++i)
	{
		alpha_mul(y[i],y[i],beta);
		// y[i] *= beta;
	}
	for(ALPHA_INT i = 1; i < m + 1; ++i)
	{
		const ALPHA_INT indx = A->pointers[i] - 1;

		ALPHA_Number v;
		v = A->values[indx];
		alpha_mul(v, v, x[i - 1]);
		alpha_madde(y[i - 1], alpha, v);
		// y[i - 1] += alpha * v * x[i - 1];
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;
 }
