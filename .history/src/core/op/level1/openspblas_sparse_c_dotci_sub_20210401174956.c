#include "openspblas/spapi.h"
#include "openspblas/kernel.h"

/*
* 
* Conjugate dot product of a compressed vector of complex numbers and real numbers
* and return the result value
*
* details:
* conjg(x[0])*y[indx[0]] + ... + conjg(x[nz-1])*y[indx[nz-1]]
*
* x         Sparse vector in compressed format
* y         full storage vector
* indx      The element index of x, stored in an array, with a length of at least nz
* nz        Number of elements in vectors x and indx
*
* input:
* x         Stored in an array, length is at least max(indx[i])
* y         Stored as an array, length is at least nz
* nz        Number of elements in vectors x and indx
* indx      The element index of x, stored in an array, with a length of at least nz
* 
* output:
* dotci		When nz>0, the result is the conjugate dot product of x and y, otherwise the value is 0
*
*/

void ONAME(const OPENSPBLAS_INT nz,
		   const OPENSPBLAS_Complex *x,
		   const OPENSPBLAS_INT *indx,
		   const OPENSPBLAS_Complex *y,
		   OPENSPBLAS_Complex *dutci)
{
	return dotci_sub(nz, x, indx, y, dutci);
}
