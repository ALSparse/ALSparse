#include "alphasparse/kernel.h"

alphasparse_status_t
ONAME(const ALPHA_INT nz,
	  const ALPHA_Number *x,
	  const ALPHA_INT *indx,
	  ALPHA_Number *y)
{
	for (ALPHA_INT i = 0; i < nz; ++i)
	{
		y[indx[i]] = x[i];
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
