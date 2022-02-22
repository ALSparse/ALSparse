#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

void ONAME(const ALPHA_INT nz,
		   const ALPHA_Complex *x,
		   const ALPHA_INT *indx,
		   const ALPHA_Complex *y,
		   ALPHA_Complex *dotci)
{
	ALPHA_Complex res;
	cmp_setzero(res);
	if (nz <= 0)
	{
		fprintf(stderr, "Invalid Values : nz <= 0 !\n");
		return;
	}
	for (ALPHA_INT i = 0; i < nz; i++)
	{
		ALPHA_Complex t;
		cmp_conj(t, x[i]);
		cmp_madde(res, t, y[indx[i]]);
	}
	*dotci = res;
	return;
}
