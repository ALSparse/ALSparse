#include "alphasparse/spapi_plain.h"
#include "alphasparse/kernel_plain.h"

void ONAME(const ALPHA_INT nz,
		   const ALPHA_Complex *x,
		   const ALPHA_INT *indx,
		   const ALPHA_Complex *y,
		   ALPHA_Complex *dutci)
{
	return dotci_sub_plain(nz, x, indx, y, dutci);
}
