#include "alphasparse/spapi.h"
#include "alphasparse/kernel.h"

/*
* 
* Add multiple scalar values ​​of a compressed vector to the full storage vector
*
* details:
* y := a*x + y
*
* a         a scalor value
* x         Sparse vector in compressed format
* y         full storage vector
*
* input:
* a         a scalor value
* x         Sparse vector in compressed format
* y         a full storage vector
* nz        Number of elements in vectors x and indx
* indx      The element index of x, stored in an array, with a length of at least nz
* 
* output:
* y         a full storage vector
*
*/

alphasparse_status_t ONAME(const ALPHA_INT nz,
                          const ALPHA_Number a,
                          const ALPHA_Number *x,
                          const ALPHA_INT *indx,
                          ALPHA_Number *y)
{
    return axpy(nz, a, x, indx, y);
}