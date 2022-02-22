#include "alphasparse/spapi.h"
#include "alphasparse/kernel.h"
/*
* 
* Gather the elements of a full storage vector into the compressed vector format
* and zero the elements at the corresponding positions in the original vector
*
* details:
* x[i] = y[indx[i]], y[indx[i]] = 0, for i=0,1,... ,nz-1
*
* x         Sparse vector in compressed format
* y         full storage vector
* indx      The element index of x, stored in an array, with a length of at least nz
*
* input:
* x         Stored in an array, length is at least max(indx[i])
* y         Stored as an array, length is at least nz
* nz        Number of elements in vectors x and indx
* indx      The element index of x, stored in an array, with a length of at least nz
* 
* output:
* x         Compressed vector x
* y         the updated y
*
*/
alphasparse_status_t ONAME(const ALPHA_INT nz,
                          ALPHA_Number *y,
                          ALPHA_Number *x,
                          const ALPHA_INT *indx)
{
    return gthrz(nz, y, x, indx);
}