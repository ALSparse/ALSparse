#include "openspblas/spapi.h"
#include "openspblas/kernel.h"

/*
* 
* Dot product of complex compressed vector and real full storage vector
*
* details:
* res = x[0]*y[indx[0]] + x[1]*y(indx[1]) +...+ x[nz - 1]*y[indx[nz - 1]]
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
* dutci		When nz>0, it contains the result of the dot product of x and y, otherwise is 0
*
*/

void ONAME(const OPENSPBLAS_INT nz,
		   const OPENSPBLAS_Complex *x,
		   const OPENSPBLAS_INT *indx,
		   const OPENSPBLAS_Complex *y,
		   OPENSPBLAS_Complex *dutci)
{
	return dotui_sub(nz, x, indx, y, dutci);
}
