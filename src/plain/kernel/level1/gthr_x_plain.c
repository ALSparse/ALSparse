#include "alphasparse/kernel_plain.h"

alphasparse_status_t
ONAME(const ALPHA_INT nz,
	  const ALPHA_Number *y,
	  ALPHA_Number *x,
	  const ALPHA_INT *indx)
{
	for (ALPHA_INT i = 0; i < nz; ++i)
	{
		x[i] = y[indx[i]];
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
