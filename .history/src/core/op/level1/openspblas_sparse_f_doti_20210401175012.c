#include "openspblas/spapi.h"
#include "openspblas/kernel.h"

/*
* 
* Dot product of compressed real number vector and full storage real number vector 
* return the result value
*
* details:
* res = x[0]*y[indx[0]] + x[1]*y[indx[1]] +...+ x[nz-1]*y[indx[nz-1]]
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
* res       When nz>0, res is the dot product result, otherwise the value is 0
*
*/

OPENSPBLAS_Number ONAME(const OPENSPBLAS_INT nz,
                 const OPENSPBLAS_Number *x,
                 const OPENSPBLAS_INT *indx,
                 const OPENSPBLAS_Number *y)
{
    return doti(nz, x, indx, y);
}