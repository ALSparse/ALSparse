#include "openspblas/spapi.h"
#include "openspblas/kernel.h"

void ONAME(const OPENSPBLAS_INT nz,
		   const OPENSPBLAS_Complex *x,
		   const OPENSPBLAS_INT *indx,
		   const OPENSPBLAS_Complex *y,
		   OPENSPBLAS_Complex *dutci)
{
	return dotci_sub(nz, x, indx, y, dutci);
}
