#include "alphasparse/spapi.h"
#include "alphasparse/kernel.h"

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

ALPHA_Number ONAME(const ALPHA_INT nz,
                 const ALPHA_Number *x,
                 const ALPHA_INT *indx,
                 const ALPHA_Number *y)
{
    return doti(nz, x, indx, y);
}