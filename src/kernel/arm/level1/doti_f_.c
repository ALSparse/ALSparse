#include "alphasparse/kernel.h"
#include <stdio.h>

ALPHA_Float ONAME(const ALPHA_INT nz,
				const ALPHA_Float *x,
				const ALPHA_INT *indx,
				const ALPHA_Float *y)
{
	ALPHA_Float res = 0.f;
	if (nz <= 0)
	{
		fprintf(stderr, "Invalid Values : nz <= 0 !\n");
		return res;
	}

	for (ALPHA_INT i = 0; i < nz; i++)
		res += x[i] * y[indx[i]];
	return res;
}
