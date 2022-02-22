#include "alphasparse/kernel.h"

alphasparse_status_t
ONAME(const ALPHA_INT nz,
	  ALPHA_Float *x,
	  const ALPHA_INT *indx,
	  ALPHA_Float *y,
	  const ALPHA_Float c,
	  const ALPHA_Float s)
{
	for (ALPHA_INT i = 0; i < nz; ++i)
	{
		ALPHA_Float t = x[i];
		x[i] = c * x[i] + s * y[indx[i]];
		y[indx[i]] = c * y[indx[i]] - s * t;
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
