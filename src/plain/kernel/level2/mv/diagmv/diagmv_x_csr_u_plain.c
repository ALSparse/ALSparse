#include "alphasparse/kernel_plain.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
	  const ALPHA_SPMAT_CSR *A,
	  const ALPHA_Number *x,
	  const ALPHA_Number beta,
	  ALPHA_Number *y)
{
    const ALPHA_INT m = A->rows;

	for(ALPHA_INT i = 0; i < m; ++i)
	{
		alpha_mule(y[i], beta);
		alpha_madde(y[i], alpha, x[i]);
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
